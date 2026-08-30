from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.recipes.matching import IngredientMatcher, normalize_ingredient, normalize_ingredients
from app.backend.recipes.ranking import RecipeRanker
from app.backend.recipes.schemas import RecipeSummary


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "recipes.db"
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(database_path))
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "rule_based")
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    initialize_database(database_path)
    return TestClient(app)




def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recipe_search_returns_matching_recipes(client: TestClient) -> None:
    response = client.post(
        "/api/recipes/search",
        json={"ingredients": ["chicken", "tomato", "egg"]},
    )

    assert response.status_code == 200
    titles = [recipe["title"] for recipe in response.json()["recipes"]]
    assert "Chicken Tomato Stir Fry" in titles
    assert "Tomato Egg Stir Fry" in titles
    assert "Chicken Omelette" in titles
    first_recipe = response.json()["recipes"][0]
    assert {
        "matched_ingredients",
        "missing_ingredients",
        "matched_count",
        "required_count",
        "coverage",
    }.issubset(first_recipe)


def test_normalize_ingredient_handles_case_whitespace_plural_and_synonyms() -> None:
    assert normalize_ingredient("Chicken") == "chicken"
    assert normalize_ingredient(" CHICKEN  ") == "chicken"
    assert normalize_ingredient("tomatoes") == "tomato"
    assert normalize_ingredient("eggs") == "egg"
    assert normalize_ingredient("capsicum") == "pepper"


def test_normalize_ingredients_removes_duplicates() -> None:
    assert normalize_ingredients(["chicken", "Chicken", " chicken "]) == ["chicken"]


def test_ingredient_matcher_calculates_coverage_and_missing_ingredients() -> None:
    matcher = IngredientMatcher()

    match = matcher.match(
        ["chicken", "tomato"],
        ["chicken", "tomato", "onion"],
    )

    assert match.matched_ingredients == ["chicken", "tomato"]
    assert match.missing_ingredients == ["onion"]
    assert match.matched_count == 2
    assert match.required_count == 3
    assert match.coverage == pytest.approx(0.667, abs=0.001)


def test_recipe_ranker_orders_by_coverage_and_applies_threshold() -> None:
    recipes = [
        RecipeSummary(id=1, title="Recipe A", matched_count=4, required_count=5, coverage=0.8),
        RecipeSummary(id=2, title="Recipe B", matched_count=3, required_count=5, coverage=0.6),
        RecipeSummary(id=3, title="Recipe C", matched_count=2, required_count=5, coverage=0.4),
    ]

    ranked = RecipeRanker().rank(recipes, minimum_coverage=0.5)

    assert [recipe.title for recipe in ranked] == ["Recipe A", "Recipe B"]


def test_recipe_detail_returns_local_recipe_data(client: TestClient) -> None:
    search_response = client.post(
        "/api/recipes/search",
        json={"ingredients": ["egg", "chicken"]},
    )
    recipe_id = next(
        recipe["id"]
        for recipe in search_response.json()["recipes"]
        if recipe["title"] == "Chicken Omelette"
    )

    response = client.get(f"/api/recipes/{recipe_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Chicken Omelette"
    assert data["cooking_time"] == 15
    assert data["difficulty"] == "easy"
    assert data["servings"] == 2
    assert data["category"] == "breakfast"
    assert data["ingredients"] == [
        {"name": "chicken", "quantity": 100, "unit": "g"},
        {"name": "egg", "quantity": 3, "unit": "pieces"},
        {"name": "onion", "quantity": 0.5, "unit": "piece"},
        {"name": "butter", "quantity": 1, "unit": "tbsp"},
    ]
    assert data["instructions"][0] == "Dice the cooked chicken and onion."


def test_recipe_detail_returns_404_for_unknown_recipe(client: TestClient) -> None:
    response = client.get("/api/recipes/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Recipe not found."}


def test_empty_ingredients_returns_no_results(client: TestClient) -> None:
    response = client.post("/api/recipes/search", json={"ingredients": []})

    assert response.status_code == 200
    assert response.json() == {"recipes": []}


def test_unknown_ingredients_return_no_results(client: TestClient) -> None:
    response = client.post("/api/recipes/search", json={"ingredients": ["durian"]})

    assert response.status_code == 200
    assert response.json() == {"recipes": []}


def test_malformed_request_returns_validation_error(client: TestClient) -> None:
    response = client.post("/api/recipes/search", json={"ingredients": "chicken"})

    assert response.status_code == 422


def test_root_serves_frontend_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<title>PantrySense</title>" in response.text
    assert "id=\"search-view\"" in response.text
    assert "id=\"ingredient-input\"" in response.text


def test_static_assets_served(client: TestClient) -> None:
    css_response = client.get("/style.css")
    assert css_response.status_code == 200
    assert "PantrySense" in css_response.text

    js_response = client.get("/app.js")
    assert js_response.status_code == 200
    assert "PantrySense" in js_response.text


def test_v030_search_behavior_regression(client: TestClient) -> None:
    response = client.post(
        "/api/recipes/search",
        json={"ingredients": ["chicken", "tomato", "egg"]},
    )

    assert response.status_code == 200
    recipes = response.json()["recipes"]
    assert len(recipes) >= 3

    # The highest coverage recipe (0.5) should be Chicken Omelette
    assert recipes[0]["title"] == "Chicken Omelette"
    assert recipes[0]["coverage"] == 0.5
    assert sorted(recipes[0]["matched_ingredients"]) == ["chicken", "egg"]
    assert sorted(recipes[0]["missing_ingredients"]) == ["butter", "onion"]

    # Subsequent recipes have coverage 0.4
    titles = [r["title"] for r in recipes]
    assert "Chicken Tomato Stir Fry" in titles
    assert "Tomato Egg Stir Fry" in titles

