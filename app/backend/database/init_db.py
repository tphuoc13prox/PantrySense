from __future__ import annotations

import json
from pathlib import Path

from app.backend.database.connection import get_connection
from app.backend.database.recipe_metadata import backfill_small_gaps, refresh_recipe_metadata
from app.backend.database.seed_data import SEED_RECIPES


def initialize_database(database_path: Path | None = None) -> None:
    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL UNIQUE,
                ingredients TEXT NOT NULL,
                ingredient_details TEXT NOT NULL DEFAULT '[]',
                instructions TEXT NOT NULL DEFAULT '[]',
                cooking_time INTEGER,
                difficulty TEXT,
                servings INTEGER,
                category TEXT
            )
            """
        )
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(recipes)")
        }
        migrations = {
            "ingredient_details": "ALTER TABLE recipes ADD COLUMN ingredient_details TEXT NOT NULL DEFAULT '[]'",
            "instructions": "ALTER TABLE recipes ADD COLUMN instructions TEXT NOT NULL DEFAULT '[]'",
            "cooking_time": "ALTER TABLE recipes ADD COLUMN cooking_time INTEGER",
            "difficulty": "ALTER TABLE recipes ADD COLUMN difficulty TEXT",
            "servings": "ALTER TABLE recipes ADD COLUMN servings INTEGER",
            "category": "ALTER TABLE recipes ADD COLUMN category TEXT",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                connection.execute(statement)

        for recipe in SEED_RECIPES:
            connection.execute(
                """
                INSERT INTO recipes (
                    title,
                    ingredients,
                    ingredient_details,
                    instructions,
                    cooking_time,
                    difficulty,
                    servings,
                    category
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(title) DO UPDATE SET
                    ingredients = excluded.ingredients,
                    ingredient_details = excluded.ingredient_details,
                    instructions = excluded.instructions,
                    cooking_time = excluded.cooking_time,
                    difficulty = excluded.difficulty,
                    servings = excluded.servings,
                    category = excluded.category
                """,
                (
                    recipe["title"],
                    json.dumps(recipe["ingredients"]),
                    json.dumps(recipe["ingredient_details"]),
                    json.dumps(recipe["instructions"]),
                    recipe["cooking_time"],
                    recipe["difficulty"],
                    recipe["servings"],
                    recipe["category"],
                ),
            )

        connection.commit()

    backfill_small_gaps(database_path)
    # Seed rows are upserted on every start; their metadata is only redone when needed.
    refresh_seed_metadata(database_path)


def refresh_seed_metadata(database_path: Path | None = None) -> int:
    """Recompute the seed recipes' metadata only when missing or stale (see refresh_recipe_metadata)."""
    return refresh_recipe_metadata(database_path, [recipe["title"] for recipe in SEED_RECIPES])
