from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.ranking.features import FEATURE_NAMES, extract_features
from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes.schemas import RecipeSummary
from scripts.train_ranker import train_ranker


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "recipes.db"
    vector_index_path = tmp_path / "recipe_vectors.index"
    vector_ids_path = tmp_path / "recipe_vector_ids.json"
    bm25_path = tmp_path / "recipe_bm25.json"
    model_path = tmp_path / "ranker_model.joblib"

    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(database_path))
    monkeypatch.setenv("PANTRYSENSE_VECTOR_INDEX_PATH", str(vector_index_path))
    monkeypatch.setenv("PANTRYSENSE_VECTOR_IDS_PATH", str(vector_ids_path))
    monkeypatch.setenv("PANTRYSENSE_BM25_INDEX_PATH", str(bm25_path))
    monkeypatch.setenv("PANTRYSENSE_RANKER_MODEL_PATH", str(model_path))
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "rule_based")
    monkeypatch.setenv("PANTRYSENSE_RANKING_MODE", "ml")
    initialize_database(database_path)
    return TestClient(app)


def test_feature_extraction_dimension() -> None:
    summary = RecipeSummary(
        id=1,
        title="Test Recipe",
        matched_ingredients=["chicken", "egg"],
        missing_ingredients=["tomato"],
        matched_count=2,
        required_count=3,
        coverage=0.667,
        semantic_score=0.85,
        bm25_score=3.5,
        rrf_score=0.031,
    )
    detail = {
        "cooking_time": 15,
        "difficulty": "easy",
        "ingredients": ["chicken", "egg", "tomato"],
    }

    features = extract_features(summary, detail)
    assert len(features) == len(FEATURE_NAMES) == 9
    assert features[0] == pytest.approx(0.85, abs=0.01)  # semantic_score
    assert features[1] == pytest.approx(3.5, abs=0.01)   # bm25_score
    assert features[2] == pytest.approx(0.031, abs=0.01) # rrf_score
    assert features[3] == pytest.approx(0.667, abs=0.01) # coverage
    assert features[4] == 2.0                            # matched_count
    assert features[5] == 1.0                            # missing_count
    assert features[6] == 15.0                           # cooking_time
    assert features[7] == 1.0                            # difficulty easy = 1.0
    assert features[8] == 3.0                            # total_ingredients


def test_ml_ranker_heuristic_fallback(tmp_path: Path) -> None:
    non_existent_model = tmp_path / "missing_ranker.joblib"
    ranker = MLRecipeRanker(model_path=non_existent_model)

    recipes = [
        RecipeSummary(id=1, title="Recipe Low", coverage=0.4, matched_count=1),
        RecipeSummary(id=2, title="Recipe High", coverage=0.9, matched_count=3),
    ]

    ranked = ranker.rank(recipes, minimum_coverage=0.3)
    assert len(ranked) == 2
    assert ranked[0].title == "Recipe High"
    assert ranked[1].title == "Recipe Low"


def test_train_and_rank_with_ml_model(tmp_path: Path) -> None:
    model_path = tmp_path / "test_ranker_model.joblib"
    train_ranker(output_path=model_path)
    assert model_path.exists()

    ranker = MLRecipeRanker(model_path=model_path)
    assert ranker.is_ready is True

    recipes = [
        RecipeSummary(
            id=1,
            title="Slow Recipe",
            coverage=0.6,
            matched_count=2,
            missing_ingredients=["onion", "garlic"],
            semantic_score=0.60,
            bm25_score=2.0,
            rrf_score=0.015,
        ),
        RecipeSummary(
            id=2,
            title="Quick Chicken Omelette",
            coverage=1.0,
            matched_count=2,
            missing_ingredients=[],
            semantic_score=0.95,
            bm25_score=5.0,
            rrf_score=0.033,
        ),
    ]
    details = {
        1: {"cooking_time": 45, "difficulty": "hard", "ingredients": ["beef", "pasta", "onion", "garlic"]},
        2: {"cooking_time": 10, "difficulty": "easy", "ingredients": ["chicken", "egg"]},
    }

    ranked = ranker.rank(recipes, details_map=details, minimum_coverage=0.3)
    assert len(ranked) == 2
    assert ranked[0].title == "Quick Chicken Omelette"
    assert ranked[0].ml_score is not None


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
