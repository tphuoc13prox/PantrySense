from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.database.recipe_metadata import (
    MetadataIndex,
    metadata_fingerprint,
    populate_recipe_metadata,
    stored_fingerprint,
)
from app.backend.recipes import dietary
from app.backend.recipes.schemas import DietaryFilters
from app.backend.recipes.service import (
    RecipeSearchService,
    _satisfies_filters,
    get_search_components,
    reset_search_components,
)
from app.backend.retrieval.bm25_store import BM25Store, preferred_index_path
from app.backend.retrieval.filtering import AllowedIds
from app.backend.retrieval.hybrid import HybridRetriever
from app.backend.retrieval.semantic import SemanticRetriever, build_recipe_representation
from app.backend.retrieval.vector_store import FaissVectorStore
from tests.test_semantic_retrieval import MockEmbedder

QUERY = ["garlic", "onion", "tomato"]
N_MEAT = 60
N_VEGAN = 10


class RankingEmbedder(MockEmbedder):
    """Scores every chicken recipe above every vegan one for the test query."""

    def embed_text(self, text: str) -> np.ndarray:
        if "chicken" in text.lower():
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "lentils" in text.lower():
            return np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _insert(conn: sqlite3.Connection, title: str, ingredients: list[str], cooking_time: int = 20) -> None:
    conn.execute(
        "INSERT INTO recipes (title, ingredients, ingredient_details, instructions, cooking_time, difficulty, servings, category) "
        "VALUES (?, ?, '[]', '[]', ?, 'easy', 2, 'dinner')",
        (title, json.dumps(ingredients), cooking_time),
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> dict[str, object]:
    """Corpus where non-vegan recipes crowd the vegan ones out of any global top-k."""
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM recipes")
        conn.execute("DELETE FROM recipe_metadata")
        for i in range(N_MEAT):
            _insert(conn, f"Garlic Tomato Chicken {i}", ["garlic", "onion", "tomato", "chicken"])
        for i in range(N_VEGAN):
            _insert(conn, f"Herb Tomato Stew {i}", ["garlic", "onion", "tomato", "basil", "thyme", "lentils", "celery"])
        conn.commit()
        rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()
    populate_recipe_metadata(db_path)

    ids = [r["id"] for r in rows]
    texts = [build_recipe_representation(r["title"], json.loads(r["ingredients"])) for r in rows]
    vegan_ids = {r["id"] for r in rows if r["title"].startswith("Herb")}

    bm25 = BM25Store()
    bm25.add_corpus(ids, texts)
    bm25_path = tmp_path / "bm25.npz"
    bm25.save(bm25_path)

    embedder = RankingEmbedder(dimension=4)
    store = FaissVectorStore(dimension=4)
    store.add(embedder.embed_texts(texts), ids)
    index_path, ids_path = tmp_path / "vec.index", tmp_path / "vec_ids.json"
    store.save(index_path, ids_path)
    semantic = SemanticRetriever(embedder=embedder, vector_store=store, index_path=index_path, ids_path=ids_path)
    hybrid = HybridRetriever(semantic_retriever=semantic, bm25_store=bm25, bm25_path=bm25_path)
    return {"db": db_path, "hybrid": hybrid, "semantic": semantic, "bm25": bm25, "vegan_ids": vegan_ids}


def _service(corpus: dict[str, object], mode: str) -> RecipeSearchService:
    hybrid = corpus["hybrid"]
    return RecipeSearchService(
        database_path=corpus["db"],
        retrieval_mode=mode,
        ranking_mode="heuristic",
        match_threshold=0.0,
        retriever=hybrid.semantic_retriever,
        hybrid_retriever=hybrid,
        bm25_store=hybrid.bm25_store,
        metadata_index=MetadataIndex(corpus["db"]),
    )


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_vegan_filter_returns_every_vegan_match(corpus: dict[str, object], mode: str) -> None:
    service = _service(corpus, mode)
    vegan_ids = corpus["vegan_ids"]

    unfiltered = service.search(QUERY)
    assert unfiltered, "query should match the corpus"
    # The setup is only meaningful if the global top-k holds no vegan recipe,
    # i.e. filtering after retrieval would have returned nothing.
    assert not {r.id for r in unfiltered} & vegan_ids

    filtered = service.search(QUERY, filters=DietaryFilters(vegan=True))
    assert {r.id for r in filtered} == vegan_ids
    assert all(r.dietary_tags["vegan"] for r in filtered)


def test_prefilter_respects_allergen_time_and_category(corpus: dict[str, object]) -> None:
    service = _service(corpus, "lexical")
    no_chicken_time = service.search(QUERY, filters=DietaryFilters(vegetarian=True, max_cooking_time=30))
    assert {r.id for r in no_chicken_time} == corpus["vegan_ids"]
    assert service.search(QUERY, category="dessert") == []
    assert service.search(QUERY, filters=DietaryFilters(vegan=True, max_cooking_time=5)) == []


def test_metadata_mask_matches_post_filter_semantics(tmp_path: Path) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)
    index = MetadataIndex(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT id, ingredients, cooking_time, category FROM recipes ORDER BY id").fetchall()

    cases = [
        (DietaryFilters(vegetarian=True), None, None),
        (DietaryFilters(gluten_free=True, dairy_free=True), None, None),
        (DietaryFilters(excluded_allergens=["Eggs", "dairy"]), None, None),
        (DietaryFilters(keto_low_carb=True, max_cooking_time=20), None, None),
        (None, 15, "break"),
        (DietaryFilters(category="dinner"), 60, "breakfast"),
    ]
    for filters, max_time, category in cases:
        mask = index.mask(filters, max_time, category)
        assert mask is not None
        for i, row in enumerate(rows):
            ings = json.loads(row["ingredients"])
            expected = _satisfies_filters(
                dietary.classify_recipe_dietary(ings),
                dietary.detect_recipe_allergens(ings),
                row["cooking_time"],
                row["category"],
                filters,
                max_time,
                category,
            )
            assert bool(mask[i]) == expected, (filters, max_time, category, row["id"])
    assert index.mask(None, None, None) is None
    assert index.mask(DietaryFilters(), None, "  ") is None


def test_bm25_npz_and_json_round_trip_give_identical_results(tmp_path: Path) -> None:
    docs = ["chicken tomato garlic", "tomato egg soup", "garlic pasta cheese", "egg egg chicken"]
    store = BM25Store()
    store.add_corpus([11, 12, 13, 14], docs)
    json_path, npz_path = tmp_path / "bm25.json", tmp_path / "bm25.npz"
    store.save(json_path)
    store.save(npz_path)

    from_json, from_npz = BM25Store(), BM25Store()
    assert from_json.load(json_path) and from_npz.load(npz_path)
    for query in ["garlic", "egg chicken", "tomato tomato soup", "unknown"]:
        assert from_json.search(query) == from_npz.search(query) == store.search(query)
    assert from_npz.doc_freqs == store.doc_freqs
    assert preferred_index_path(json_path) == npz_path


def test_bm25_allowed_restricts_results(tmp_path: Path) -> None:
    store = BM25Store()
    store.add_corpus([1, 2, 3], ["garlic garlic", "garlic onion", "garlic"])
    allowed = AllowedIds.from_ids([3])
    assert [r[0] for r in store.search("garlic", allowed=allowed)] == [3]
    assert store.search("garlic", allowed=AllowedIds.from_ids([])) == []


def test_faiss_selector_returns_exact_top_k_within_allowed() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(500, 16)).astype(np.float32)
    ids = list(range(1000, 1500))
    store = FaissVectorStore(dimension=16)
    store.add(vectors, ids)
    allowed_ids = [i for i in ids if i % 7 == 0]
    allowed = AllowedIds.from_ids(allowed_ids)
    query = rng.normal(size=16).astype(np.float32)

    results = store.search(query, top_k=10, min_threshold=-1.0, allowed=allowed)
    assert len(results) == 10
    assert all(r[0] % 7 == 0 for r in results)

    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = normed @ (query / np.linalg.norm(query))
    expected = sorted(allowed_ids, key=lambda rid: -scores[rid - 1000])[:10]
    assert [r[0] for r in results] == expected


def test_allowed_ids_maps_to_any_store_order() -> None:
    allowed = AllowedIds(np.array([1, 2, 3, 4], dtype=np.int64), np.array([True, False, True, False]))
    order = np.array([4, 3, 9, 1], dtype=np.int64)
    assert allowed.mask_for(order).tolist() == [False, True, False, True]
    assert 3 in allowed and 2 not in allowed and 9 not in allowed


def test_services_share_components_per_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_search_components()
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)
    monkeypatch.setenv("PANTRYSENSE_BM25_INDEX_PATH", str(tmp_path / "a.json"))
    first = RecipeSearchService(database_path=db_path, retrieval_mode="lexical")
    second = RecipeSearchService(database_path=db_path, retrieval_mode="hybrid")
    assert first.hybrid_retriever is second.hybrid_retriever
    assert first.bm25_store is first.hybrid_retriever.bm25_store

    monkeypatch.setenv("PANTRYSENSE_BM25_INDEX_PATH", str(tmp_path / "b.json"))
    third = RecipeSearchService(database_path=db_path)
    assert third.hybrid_retriever is not first.hybrid_retriever
    assert get_search_components(db_path) is get_search_components(db_path)
    reset_search_components()


def test_initialize_database_fills_metadata_and_fingerprint(tmp_path: Path) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        stored = conn.execute("SELECT COUNT(*) FROM recipe_metadata").fetchone()[0]
        assert stored == total > 0
    populate_recipe_metadata(db_path)
    with get_connection(db_path) as conn:
        assert stored_fingerprint(conn) == metadata_fingerprint()


def test_search_omits_substitutions_unless_requested(tmp_path: Path) -> None:
    db_path = tmp_path / "recipes.db"
    initialize_database(db_path)
    service = RecipeSearchService(database_path=db_path, retrieval_mode="rule_based", match_threshold=0.3)
    plain = service.search(["chicken", "tomato", "egg"])
    with_subs = service.search(["chicken", "tomato", "egg"], include_substitutions=True)
    assert plain and all(r.substitutions == [] for r in plain)
    assert any(r.substitutions for r in with_subs)
    assert all(r.nutrition is not None and r.nutrition.per_serving.calories > 0 for r in plain)
