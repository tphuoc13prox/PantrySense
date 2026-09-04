from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.recipes.service import RecipeSearchService
from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.semantic import (
    CandidateMatch,
    SemanticIndexNotFoundError,
    SemanticRetriever,
    build_query_representation,
    build_recipe_representation,
)
from app.backend.retrieval.vector_store import FaissVectorStore


class MockEmbedder(IngredientEmbedder):
    """Deterministic test double that maps keyword strings to orthogonal unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.model_name = "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        text_lower = text.lower()
        if "chicken" in text_lower or "omelette" in text_lower:
            vec[0] = 1.0
        elif "tomato" in text_lower or "egg" in text_lower:
            vec[1] = 1.0
        elif "pasta" in text_lower:
            vec[2] = 1.0
        else:
            vec[3] = 1.0
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)


def test_representations_are_deterministic() -> None:
    query = build_query_representation(["  Chicken  ", "Tomato", "chicken", "eggs"])
    assert query == "chicken egg tomato"

    recipe_repr = build_recipe_representation("Chicken Omelette", ["chicken", "egg", "butter"])
    assert recipe_repr == "Chicken Omelette\nbutter chicken egg"


def test_vector_store_add_search_save_load(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=4)
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    ids = [101, 102, 103]

    store.add(vectors, ids)
    assert store.total_vectors == 3

    # Search for vector [1.0, 0.0, 0.0, 0.0]
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2, min_threshold=0.5)

    assert len(results) == 1
    assert results[0][0] == 101
    assert results[0][1] == pytest.approx(1.0, abs=0.01)
    assert results[0][2] == 1  # Rank 1

    # Test Save & Load
    index_path = tmp_path / "test.index"
    ids_path = tmp_path / "test_ids.json"

    store.save(index_path, ids_path)
    assert ids_path.exists()

    new_store = FaissVectorStore(dimension=4)
    loaded = new_store.load(index_path, ids_path)
    assert loaded is True
    assert new_store.total_vectors == 3
    assert new_store.recipe_ids == [101, 102, 103]


def test_semantic_retriever_with_mock_embedder(tmp_path: Path) -> None:
    embedder = MockEmbedder(dimension=4)
    vector_store = FaissVectorStore(dimension=4)

    # Add 2 recipes
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Recipe 1 (Chicken)
        [0.0, 1.0, 0.0, 0.0],  # Recipe 2 (Tomato Egg)
    ], dtype=np.float32)
    vector_store.add(embeddings, [1, 2])

    index_path = tmp_path / "recipe_vectors.index"
    ids_path = tmp_path / "recipe_vector_ids.json"
    vector_store.save(index_path, ids_path)

    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        index_path=index_path,
        ids_path=ids_path,
    )

    candidates = retriever.retrieve(["chicken"], top_k=5, min_threshold=0.5)
    assert len(candidates) >= 1
    assert candidates[0].recipe_id == 1
    assert candidates[0].semantic_score == pytest.approx(1.0, abs=0.01)


def test_missing_index_raises_semantic_index_not_found(tmp_path: Path) -> None:
    non_existent_index = tmp_path / "non_existent.index"
    non_existent_ids = tmp_path / "non_existent.json"

    retriever = SemanticRetriever(
        index_path=non_existent_index,
        ids_path=non_existent_ids,
    )

    with pytest.raises(SemanticIndexNotFoundError, match="Semantic vector index not found"):
        retriever.retrieve(["chicken"])


def test_search_service_rule_based_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)

    service = RecipeSearchService(
        database_path=db_path,
        retrieval_mode="rule_based",
        match_threshold=0.3,
    )

    results = service.search(["chicken", "tomato", "egg"])
    assert len(results) >= 3
    titles = [r.title for r in results]
    assert "Chicken Omelette" in titles
    assert "Chicken Tomato Stir Fry" in titles


def test_api_returns_503_when_semantic_index_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)

    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(db_path))
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "semantic")
    monkeypatch.setenv("PANTRYSENSE_VECTOR_INDEX_PATH", str(tmp_path / "missing.index"))
    monkeypatch.setenv("PANTRYSENSE_VECTOR_IDS_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_BUILD_INDEX", "false")



    client = TestClient(app)
    response = client.post(
        "/api/recipes/search",
        json={"ingredients": ["chicken"]},
    )

    assert response.status_code == 503
    assert "Semantic vector index not found" in response.json()["detail"]
