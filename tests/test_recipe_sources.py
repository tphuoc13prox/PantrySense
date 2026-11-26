"""recipe_source rows reach the stored dietary tags through the migration path."""
from __future__ import annotations

import json
from pathlib import Path

from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.database.recipe_metadata import RecipeSource, populate_recipe_metadata, store_recipe_sources
from app.backend.recipes.service import RecipeSearchService


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "recipes.db"
    initialize_database(path)
    with get_connection(path) as conn:
        conn.execute("DELETE FROM recipes")
        for title in ("Herb Rub Roast", "Garden Salad", "Curated Salad"):
            conn.execute(
                "INSERT INTO recipes (title, ingredients, ingredient_details, instructions, cooking_time, "
                "difficulty, servings, category) VALUES (?, ?, '[]', '[]', 30, 'easy', 4, 'dinner')",
                (title, json.dumps(["olive oil", "rosemary", "garlic"])),
            )
        conn.commit()
    return path


def test_sources_change_stored_tags(tmp_path: Path) -> None:
    path = _db(tmp_path)
    sources = {
        "herb rub roast": RecipeSource(keywords=["Easy"], category="Roast Beef"),
        "garden salad": RecipeSource(keywords=["Vegetable"], extra_ingredients=["parmesan cheese"]),
        "curated salad": RecipeSource(keywords=["Meat"]),
    }
    assert store_recipe_sources(path, sources, skip_titles=["Curated Salad"]) == 2
    populate_recipe_metadata(path)
    with get_connection(path) as conn:
        rows = conn.execute(
            "SELECT r.id, r.title, m.vegetarian, m.vegan, m.allergens FROM recipes r "
            "JOIN recipe_metadata m ON m.recipe_id = r.id"
        ).fetchall()
    got = {r["title"]: (r["vegetarian"], r["vegan"], json.loads(r["allergens"])) for r in rows}
    assert got["Herb Rub Roast"][:2] == (0, 0)
    assert got["Garden Salad"] == (1, 0, ["dairy"])
    assert got["Curated Salad"][:2] == (1, 1)

    # The detail view computes the same tags as the stored ones.
    service = RecipeSearchService(database_path=path)
    for row in rows:
        detail = service.get_by_id(row["id"])
        assert detail is not None
        assert bool(detail.dietary_tags["vegetarian"]) == bool(row["vegetarian"])
        assert bool(detail.dietary_tags["vegan"]) == bool(row["vegan"])
