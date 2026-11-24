from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.ranking.features import (
    ABSENT_RANK,
    DEFAULT_COOKING_TIME,
    FEATURE_NAMES,
    LTR_FEATURE_NAMES,
    build_feature_matrix,
    extract_features,
)
from app.backend.ranking.metrics import hit_at_k, mrr_at_k, ndcg_at_k
from app.backend.ranking.ml_ranker import BUNDLE_FORMAT, MLRecipeRanker
from app.backend.recipes.schemas import RecipeSummary
from scripts.build_ltr_dataset import EVAL_BUCKETS, QueryGroup, bucket_of, generate_queries
from scripts.train_ranker import train_from_groups


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "recipes.db"
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(database_path))
    monkeypatch.setenv("PANTRYSENSE_VECTOR_INDEX_PATH", str(tmp_path / "recipe_vectors.index"))
    monkeypatch.setenv("PANTRYSENSE_VECTOR_IDS_PATH", str(tmp_path / "recipe_vector_ids.json"))
    monkeypatch.setenv("PANTRYSENSE_BM25_INDEX_PATH", str(tmp_path / "recipe_bm25.json"))
    monkeypatch.setenv("PANTRYSENSE_RANKER_MODEL_PATH", str(tmp_path / "ranker_model.joblib"))
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "rule_based")
    monkeypatch.setenv("PANTRYSENSE_RANKING_MODE", "ml")
    initialize_database(database_path)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _ml_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PANTRYSENSE_RANKING_MODE", "ml")


def _summary(rid: int, title: str, matched: list[str], missing: list[str], **kw) -> RecipeSummary:
    req = len(matched) + len(missing)
    return RecipeSummary(
        id=rid,
        title=title,
        matched_ingredients=matched,
        missing_ingredients=missing,
        matched_count=len(matched),
        required_count=req,
        coverage=len(matched) / req if req else 0.0,
        **kw,
    )


def test_legacy_feature_extraction_dimension() -> None:
    summary = RecipeSummary(
        id=1, title="Test Recipe", matched_ingredients=["chicken", "egg"], missing_ingredients=["tomato"],
        matched_count=2, required_count=3, coverage=0.667, semantic_score=0.85, bm25_score=3.5, rrf_score=0.031,
    )
    detail = {"cooking_time": 15, "difficulty": "easy", "ingredients": ["chicken", "egg", "tomato"]}
    features = extract_features(summary, detail)
    assert len(features) == len(FEATURE_NAMES) == 9
    assert features[0] == pytest.approx(0.85)
    assert features[6] == 15.0
    assert features[7] == 1.0
    assert features[8] == 3.0


def test_build_feature_matrix_query_relative() -> None:
    recipes = [
        _summary(1, "Chicken Garlic Rice", ["chicken", "garlic"], ["rice", "salt"],
                 semantic_score=0.8, bm25_score=4.0, rrf_score=0.03, retrieval_rank=1),
        _summary(2, "Garlic Bread", ["garlic"], ["bread", "butter"],
                 semantic_score=None, bm25_score=2.0, rrf_score=0.016, retrieval_rank=2),
        _summary(3, "Plain Soup", ["chicken"], ["carrot"],
                 semantic_score=0.6, bm25_score=None, rrf_score=0.015, retrieval_rank=3),
    ]
    details = {1: {"cooking_time": 30, "difficulty": "easy"}, 2: {"cooking_time": None, "difficulty": "medium"}}
    X = build_feature_matrix(recipes, details)
    f = {name: X[:, i] for i, name in enumerate(LTR_FEATURE_NAMES)}

    assert X.shape == (3, len(LTR_FEATURE_NAMES))
    assert list(f["has_dense"]) == [1.0, 0.0, 1.0]
    assert list(f["has_bm25"]) == [1.0, 1.0, 0.0]
    assert list(f["bm25_rank"]) == [1.0, 2.0, ABSENT_RANK]
    assert list(f["dense_rank"]) == [1.0, ABSENT_RANK, 2.0]
    assert f["bm25_rel"][0] == pytest.approx(1.0) and f["bm25_rel"][1] == pytest.approx(0.5)
    assert f["dense_gap"][2] == pytest.approx(0.2)
    # query approximated by the union of matched ingredients: {chicken, garlic}
    assert set(f["query_size"]) == {2.0}
    assert list(f["query_coverage"]) == [1.0, 0.5, 0.5]
    # salt and butter are staples, so they do not count as missing
    assert list(f["missing_nonstaple_count"]) == [1.0, 1.0, 1.0]
    assert f["title_overlap_count"][0] == 2.0
    assert f["title_overlap"][2] == 0.0
    assert f["cooking_time"][1] == DEFAULT_COOKING_TIME and f["cooking_time_missing"][1] == 1.0
    assert f["cooking_time"][0] == 30.0 and f["cooking_time_missing"][0] == 0.0
    assert build_feature_matrix([], {}).shape == (0, len(LTR_FEATURE_NAMES))


def test_ranking_metrics() -> None:
    assert ndcg_at_k([2, 0, 0], 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 2], 10) == pytest.approx(1.0 / np.log2(3))
    assert ndcg_at_k([0, 0], 10) == 0.0
    assert mrr_at_k([0, 0, 1], 10) == pytest.approx(1 / 3)
    assert mrr_at_k([0] * 10 + [2], 10) == 0.0
    assert hit_at_k([0, 2], 1) == 0.0 and hit_at_k([2], 1) == 1.0


def test_ml_ranker_heuristic_fallback(tmp_path: Path) -> None:
    ranker = MLRecipeRanker(model_path=tmp_path / "missing_ranker.joblib")
    recipes = [
        RecipeSummary(id=1, title="Recipe Low", coverage=0.4, matched_count=1),
        RecipeSummary(id=2, title="Recipe High", coverage=0.9, matched_count=3),
        RecipeSummary(id=3, title="Recipe Filtered", coverage=0.1, matched_count=1),
    ]
    ranked = ranker.rank(recipes, minimum_coverage=0.3)
    assert [r.title for r in ranked] == ["Recipe High", "Recipe Low"]
    assert all(r.ml_score is None for r in ranked)


def test_bundle_with_unknown_feature_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump({"format": BUNDLE_FORMAT, "model": object(), "feature_names": ["nope"]}, path)
    ranker = MLRecipeRanker(model_path=path)
    assert ranker.is_ready is False
    ranked = ranker.rank([RecipeSummary(id=1, title="A", coverage=0.5, matched_count=1)], minimum_coverage=0.1)
    assert len(ranked) == 1 and ranked[0].ml_score is None


def test_legacy_bare_model_still_loads(tmp_path: Path) -> None:
    from sklearn.linear_model import LinearRegression

    X = np.array([[0.9, 5, 0.03, 1.0, 2, 0, 10, 1, 2], [0.1, 0, 0.01, 0.2, 1, 4, 60, 2, 5]], dtype=np.float32)
    path = tmp_path / "legacy.joblib"
    joblib.dump(LinearRegression().fit(X, [3.0, 0.0]), path)
    ranker = MLRecipeRanker(model_path=path)
    recipes = [
        RecipeSummary(id=1, title="Weak", coverage=0.2, matched_count=1, missing_ingredients=list("abcd"),
                      semantic_score=0.1, bm25_score=0.0, rrf_score=0.01),
        RecipeSummary(id=2, title="Strong", coverage=1.0, matched_count=2, semantic_score=0.9, bm25_score=5.0,
                      rrf_score=0.03),
    ]
    ranked = ranker.rank(recipes, minimum_coverage=0.1)
    assert ranked[0].title == "Strong" and ranked[0].ml_score is not None


def _synthetic_groups(n_queries: int, seed: int = 0) -> list[QueryGroup]:
    """Queries where the relevant recipe uses all query ingredients but is not first by coverage."""
    rng = random.Random(seed)
    vocab = [f"ing{i}" for i in range(40)]
    groups = []
    for q in range(n_queries):
        query = rng.sample(vocab, 3)
        recipes, labels = [], []
        for c in range(12):
            rid = q * 100 + c
            if c == 0:
                matched, missing = list(query), [f"x{j}" for j in range(rng.randint(4, 7))]
            else:
                matched = rng.sample(query, rng.randint(1, 2))
                missing = [f"y{j}" for j in range(rng.randint(0, 3))]
            recipes.append(_summary(
                rid, f"dish {rid}", matched, missing,
                semantic_score=rng.uniform(0.4, 0.9), bm25_score=rng.uniform(1, 6),
                rrf_score=rng.uniform(0.01, 0.03), retrieval_rank=c + 1,
            ))
            labels.append(2 if c == 0 else 0)
        order = list(range(12))
        rng.shuffle(order)
        groups.append(QueryGroup(
            qid=f"q{q}", recipes=[recipes[i] for i in order], details_map={},
            labels=[labels[i] for i in order], passes=[True] * 12, threshold=0.0,
        ))
    return groups


def test_train_and_rank_with_ml_model(tmp_path: Path) -> None:
    model_path = tmp_path / "test_ranker_model.joblib"
    grid = {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [5]}
    report = train_from_groups(_synthetic_groups(60), model_path, n_folds=3, grid=grid, ablation=False)
    assert model_path.exists()
    assert report["cv_conditional"]["lambdamart"]["ndcg@10"] > report["cv_conditional"]["heuristic"]["ndcg@10"]

    bundle = joblib.load(model_path)
    assert bundle["format"] == BUNDLE_FORMAT and bundle["feature_names"] == LTR_FEATURE_NAMES

    ranker = MLRecipeRanker(model_path=model_path)
    assert ranker.is_ready is True
    test_group = _synthetic_groups(1, seed=99)[0]
    ranked = ranker.rank(test_group.recipes, details_map={}, minimum_coverage=0.0)
    assert len(ranked) == 12
    assert all(r.ml_score is not None for r in ranked)
    assert [r.ml_score for r in ranked] == sorted((r.ml_score for r in ranked), reverse=True)
    relevant_id = next(r.id for r, lb in zip(test_group.recipes, test_group.labels) if lb == 2)
    assert relevant_id in [r.id for r in ranked[:3]]


def test_query_generation_uses_training_pool_only() -> None:
    rng = random.Random(1)
    vocab = [f"ingredient{i}" for i in range(200)]
    corpus = [
        (rid, f"Recipe {rid} {''.join(rng.choice('abcdefgh') for _ in range(6))}", tuple(sorted(rng.sample(vocab, 6))))
        for rid in range(1, 600)
    ]
    queries, meta = generate_queries(corpus, n_train=20, n_heldout=5, seed=3)
    assert len(queries) == 25
    for q in queries:
        assert bucket_of(q.recipe_id) not in EVAL_BUCKETS
        assert (bucket_of(q.recipe_id) == 9) == (q.split == "heldout")
        assert 2 <= len(q.ingredients) <= 7
        assert q.recipe_id in q.relevant_ids
    again, _ = generate_queries(corpus, n_train=20, n_heldout=5, seed=3)
    assert [q.ingredients for q in again] == [q.ingredients for q in queries]

    used = frozenset(i for q in queries for i in q.relevant_ids)
    pantry, _ = generate_queries(
        corpus, n_train=20, n_heldout=5, seed=4, tiers=("pantry",), exclude_ids=used, qid_prefix="p", extra_min_df=1,
    )
    source = {rid: set(ings) for rid, _, ings in corpus}
    for q in pantry:
        assert q.tier == "pantry" and q.qid[1] == "p"
        assert q.recipe_id not in used
        assert 2 <= len(q.extras) <= 4
        assert set(q.extras) <= set(q.ingredients)
        assert not set(q.extras) & source[q.recipe_id]
        assert not set(q.ingredients) <= source[q.recipe_id]


def test_system_setup_status_api(client: TestClient) -> None:
    res = client.get("/api/system/setup-status")
    assert res.status_code == 200
    data = res.json()
    assert "is_ready" in data
    assert "has_dataset" in data
    assert "recipe_count" in data


def test_reset_dataset_api(client: TestClient) -> None:
    res = client.post("/api/system/reset-dataset")
    assert res.status_code == 200
    assert res.json()["success"] is True

    status_res = client.get("/api/system/setup-status")
    assert status_res.status_code == 200
    assert status_res.json()["is_ready"] is False
    assert status_res.json()["recipe_count"] == 0
