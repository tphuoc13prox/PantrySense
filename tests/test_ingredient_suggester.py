from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.recipes.suggester import IngredientSuggester, get_ingredient_suggester


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "recipes.db"
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(database_path))
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "rule_based")
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    initialize_database(database_path)
    # Reset cached singleton for clean testing
    get_ingredient_suggester(db_path=database_path, force_reload=True)
    return TestClient(app)


def test_suggester_with_custom_vocabulary() -> None:
    vocab = {
        "chicken breast": 500,
        "chicken thigh": 300,
        "cheddar cheese": 200,
        "parmesan cheese": 150,
        "garlic clove": 400,
        "garlic powder": 100,
        "tomato": 600,
        "tomato paste": 250,
    }
    suggester = IngredientSuggester(vocab_freq=vocab)

    # Prefix match
    prefix_results = suggester.suggest("chick", limit=5)
    assert len(prefix_results) == 2
    assert prefix_results[0].name == "chicken breast"
    assert prefix_results[1].name == "chicken thigh"
    assert not prefix_results[0].is_correction

    # Substring match
    substr_results = suggester.suggest("cheese", limit=5)
    assert len(substr_results) == 2
    names = [r.name for r in substr_results]
    assert "cheddar cheese" in names
    assert "parmesan cheese" in names

    # Typo / Fuzzy correction
    fuzzy_results = suggester.suggest("chikcen", limit=5)
    assert len(fuzzy_results) > 0
    assert "chicken" in fuzzy_results[0].name
    assert fuzzy_results[0].is_correction is True

    # Typo for tomato
    typo_tomato = suggester.suggest("tomto", limit=5)
    assert len(typo_tomato) > 0
    assert typo_tomato[0].name == "tomato"
    assert typo_tomato[0].is_correction is True


def test_suggester_empty_query() -> None:
    vocab = {"sugar": 100, "salt": 500, "pepper": 300}
    suggester = IngredientSuggester(vocab_freq=vocab)
    results = suggester.suggest("", limit=2)
    assert len(results) == 2
    assert results[0].name == "salt"
    assert results[1].name == "pepper"


def test_api_ingredients_suggest_endpoint(client: TestClient) -> None:
    # Test prefix query
    response = client.get("/api/ingredients/suggest?q=chi&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "suggestions" in data
    assert data["query"] == "chi"
    assert len(data["suggestions"]) > 0
    assert any("chicken" in s["name"].lower() for s in data["suggestions"])


def test_api_ingredients_suggest_fuzzy_typo(client: TestClient) -> None:
    # Test typo correction
    response = client.get("/api/ingredients/suggest?q=chikcen&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert len(data["suggestions"]) > 0
    first = data["suggestions"][0]
    assert "chicken" in first["name"].lower()
    assert first["is_correction"] is True


def test_api_ingredients_suggest_limit(client: TestClient) -> None:
    response = client.get("/api/ingredients/suggest?q=e&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["suggestions"]) <= 2


def test_normalize_ingredient_audited_rules() -> None:
    from app.backend.recipes.matching import normalize_ingredient

    # Egg tomato -> roma tomato
    assert normalize_ingredient("egg tomato") == "roma tomato"
    assert normalize_ingredient("egg tomatoes") == "roma tomato"
    assert normalize_ingredient("plum tomatoes") == "roma tomato"

    # Dialects
    assert normalize_ingredient("capsicum") == "pepper"
    assert normalize_ingredient("aubergine") == "eggplant"
    assert normalize_ingredient("courgette") == "zucchini"
    assert normalize_ingredient("spring onions") == "green onion"
    assert normalize_ingredient("scallions") == "green onion"
    assert normalize_ingredient("coriander") == "cilantro"

    # Trailing prepositions and stopwords
    assert normalize_ingredient("diced tomatoes with") == "diced tomato"
    assert normalize_ingredient("lemon juice and") == "lemon juice"
    assert normalize_ingredient("green onions with") == "green onion"
    assert normalize_ingredient("white tuna in") == "tuna"

    # Leading measures & fat percentage
    assert normalize_ingredient("2 low-fat milk") == "milk"
    assert normalize_ingredient("8-inch flour tortilla") == "flour tortilla"
    assert normalize_ingredient("kraft 100 parmesan") == "parmesan cheese"
