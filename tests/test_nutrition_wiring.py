"""Stored ingredient quantities reach the nutrition estimate in search and detail paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.database.recipe_metadata import compute_recipe_metadata, populate_recipe_metadata
from app.backend.recipes import nutrition
from app.backend.recipes.service import RecipeSearchService

INGREDIENTS = ["flour", "butter", "sugar"]
SMALL = [
    {"name": "flour", "quantity": 0.5, "unit": "cup"},
    {"name": "butter", "quantity": 1.0, "unit": "tablespoon"},
    {"name": "sugar", "quantity": 1.0, "unit": "tablespoon"},
]
LARGE = [
    {"name": "flour", "quantity": 4.0, "unit": "cup"},
    {"name": "butter", "quantity": 1.0, "unit": "cup"},
    {"name": "sugar", "quantity": 2.0, "unit": "cup"},
]


def test_compute_recipe_metadata_uses_quantities() -> None:
    small = compute_recipe_metadata(INGREDIENTS, 4, SMALL)
    large = compute_recipe_metadata(INGREDIENTS, 4, LARGE)
    names_only = compute_recipe_metadata(INGREDIENTS, 4)

    assert large.total["calories"] > 5 * small.total["calories"]
    expected = nutrition.calculate_recipe_nutrition(INGREDIENTS, servings=4, ingredient_details=LARGE)
    assert large.total["calories"] == pytest.approx(expected["total"]["calories"])
    # Dietary tags and allergens come from names, so quantities do not change them.
    assert small.dietary_tags == large.dietary_tags == names_only.dietary_tags
    assert small.allergens == large.allergens == names_only.allergens


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "recipes.db"
    initialize_database(path)
    with get_connection(path) as conn:
        conn.execute("DELETE FROM recipes")
        for title, details in (("Small Cake", SMALL), ("Large Cake", LARGE)):
            conn.execute(
                "INSERT INTO recipes (title, ingredients, ingredient_details, instructions, cooking_time, "
                "difficulty, servings, category) VALUES (?, ?, ?, '[]', 30, 'easy', 4, 'dessert')",
                (title, json.dumps(INGREDIENTS), json.dumps(details)),
            )
        conn.commit()
    return path


def test_migration_and_detail_agree_on_quantity_aware_nutrition(db_path: Path) -> None:
    populate_recipe_metadata(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT r.id, r.title, m.calories FROM recipes r JOIN recipe_metadata m ON m.recipe_id = r.id"
        ).fetchall()
    stored = {r["title"]: (r["id"], r["calories"]) for r in rows}
    assert stored["Large Cake"][1] > 5 * stored["Small Cake"][1]

    service = RecipeSearchService(database_path=db_path)
    for title, details in (("Small Cake", SMALL), ("Large Cake", LARGE)):
        recipe_id, stored_calories = stored[title]
        detail = service.get_by_id(recipe_id)
        assert detail is not None
        expected = nutrition.calculate_recipe_nutrition(INGREDIENTS, servings=4, ingredient_details=details)
        assert detail.nutrition.per_serving.calories == pytest.approx(expected["per_serving"]["calories"], abs=0.1)
        assert stored_calories == pytest.approx(detail.nutrition.per_serving.calories, abs=0.1)
