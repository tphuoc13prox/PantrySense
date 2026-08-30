from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from app.backend.database.init_db import initialize_database
from app.backend.recipes.service import RecipeSearchService
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.fusion import reciprocal_rank_fusion
from app.backend.retrieval.hybrid import HybridRetriever
from app.backend.retrieval.semantic import SemanticRetriever
from app.backend.retrieval.vector_store import FaissVectorStore
from tests.test_semantic_retrieval import MockEmbedder


def test_bm25_store_index_and_search(tmp_path: Path) -> None:
    store = BM25Store()
    docs = [
        "Chicken tomato garlic stir fry",
        "Tomato egg soup with green onion",
        "Creamy garlic pasta with cheese",
    ]
    ids = [101, 102, 103]

    store.add_corpus(ids, docs)
    assert store.total_docs == 3

    # Search for "garlic" -> should match 101 and 103
    results_garlic = store.search("garlic", top_k=5)
    garlic_ids = [r[0] for r in results_garlic]
    assert 101 in garlic_ids
    assert 103 in garlic_ids
    assert 102 not in garlic_ids

    # Search for "soup" -> should match 102
    results_soup = store.search("soup", top_k=5)
    assert len(results_soup) == 1
    assert results_soup[0][0] == 102

    # Test save and load
    save_path = tmp_path / "test_bm25.json"
    store.save(save_path)
    assert save_path.exists()

    new_store = BM25Store()
    assert new_store.load(save_path) is True
    assert new_store.total_docs == 3
    assert new_store.recipe_ids == [101, 102, 103]


def test_reciprocal_rank_fusion_math() -> None:
    dense_results = [
        (101, 0.95, 1),
        (102, 0.85, 2),
        (103, 0.70, 3),
    ]
    sparse_results = [
        (102, 4.5, 1),
        (101, 3.2, 2),
        (104, 2.1, 3),
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=5)

    # 101: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    # 102: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522
    # 103: 1/(60+3) = 1/63 = 0.015873
    # 104: 1/(60+3) = 1/63 = 0.015873

    top_ids = [c.recipe_id for c in fused]
    assert top_ids[:2] == [101, 102] or top_ids[:2] == [102, 101]
    assert fused[0].rrf_score == pytest.approx(0.032522, abs=0.0001)
    assert fused[0].semantic_score is not None
    assert fused[0].bm25_score is not None


def test_hybrid_retriever_flow(tmp_path: Path) -> None:
    # 1. Setup Mock Dense Retriever
    embedder = MockEmbedder(dimension=4)
    vector_store = FaissVectorStore(dimension=4)
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 1: Chicken
        [0.0, 1.0, 0.0, 0.0],  # 2: Tomato
    ], dtype=np.float32)
    vector_store.add(embeddings, [1, 2])

    sem_index_path = tmp_path / "recipe_vectors.index"
    sem_ids_path = tmp_path / "recipe_vector_ids.json"
    vector_store.save(sem_index_path, sem_ids_path)

    sem_retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        index_path=sem_index_path,
        ids_path=sem_ids_path,
    )

    # 2. Setup BM25 Store
    bm25_store = BM25Store()
    bm25_store.add_corpus([1, 2], ["chicken omelette with egg", "tomato soup with egg"])
    bm25_path = tmp_path / "recipe_bm25.json"
    bm25_store.save(bm25_path)

    # 3. Hybrid Retriever
    hybrid = HybridRetriever(
        semantic_retriever=sem_retriever,
        bm25_store=bm25_store,
        bm25_path=bm25_path,
    )

    candidates = hybrid.retrieve(["chicken", "egg"], top_k=5)
    assert len(candidates) >= 1
    assert candidates[0].recipe_id == 1
    assert candidates[0].rrf_score is not None


def test_service_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)

    # Test rule_based mode
    service_rule = RecipeSearchService(
        database_path=db_path,
        retrieval_mode="rule_based",
        match_threshold=0.3,
    )
    res_rule = service_rule.search(["chicken", "tomato"])
    assert len(res_rule) >= 1
