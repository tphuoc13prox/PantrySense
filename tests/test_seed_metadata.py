"""Seed recipes get the same metadata as the migration writes, and only when needed."""
from __future__ import annotations

import json
from pathlib import Path

from app.backend.database import init_db, recipe_metadata
from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.database.recipe_metadata import (
    FINGERPRINT_KEY,
    compute_recipe_metadata,
    populate_recipe_metadata,
)
from app.backend.database.seed_data import SEED_RECIPES


def _stored(path: Path) -> dict[str, tuple]:
    with get_connection(path) as conn:
        rows = conn.execute(
            "SELECT r.title, m.* FROM recipes r JOIN recipe_metadata m ON m.recipe_id = r.id"
        ).fetchall()
    return {row["title"]: tuple(row)[1:] for row in rows}


def test_seed_metadata_matches_migration(tmp_path: Path) -> None:
    path = tmp_path / "recipes.db"
    initialize_database(path)
    at_start = _stored(path)
    populate_recipe_metadata(path)
    assert _stored(path) == at_start

    # Quantities reach the seed nutrition, as they do in the migration.
    seed = SEED_RECIPES[0]
    with_details = compute_recipe_metadata(seed["ingredients"], seed["servings"], seed["ingredient_details"],
                                           title=seed["title"])
    with get_connection(path) as conn:
        row = conn.execute(
            "SELECT m.calories FROM recipes r JOIN recipe_metadata m ON m.recipe_id = r.id WHERE r.title = ?",
            (seed["title"],),
        ).fetchone()
    assert row["calories"] == with_details.per_serving["calories"]


def test_seed_metadata_is_not_recomputed_on_every_start(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "recipes.db"
    initialize_database(path)

    calls: list[int] = []
    real = init_db.refresh_recipe_metadata

    def counting(database_path, titles):
        written = real(database_path, titles)
        calls.append(written)
        return written

    monkeypatch.setattr(init_db, "refresh_recipe_metadata", counting)
    initialize_database(path)
    assert calls == [0]

    # Act like the full corpus, where the inline backfill does not run.
    monkeypatch.setattr(recipe_metadata, "INLINE_BACKFILL_LIMIT", 0)

    # A missing row is filled in.
    with get_connection(path) as conn:
        conn.execute(
            "DELETE FROM recipe_metadata WHERE recipe_id = (SELECT id FROM recipes WHERE title = ?)",
            (SEED_RECIPES[0]["title"],),
        )
        conn.commit()
    initialize_database(path)
    assert calls[-1] == 1

    # A fingerprint from other estimator code redoes every seed row.
    with get_connection(path) as conn:
        conn.execute("UPDATE recipe_metadata_info SET value = 'old' WHERE key = ?", (FINGERPRINT_KEY,))
        conn.execute("UPDATE recipe_metadata SET calories = -1")
        conn.commit()
    initialize_database(path)
    assert calls[-1] == len(SEED_RECIPES)
    assert all(row[-6] >= 0 for row in _stored(path).values())


def test_seed_upsert_keeps_user_tables(tmp_path: Path) -> None:
    path = tmp_path / "recipes.db"
    initialize_database(path)
    with get_connection(path) as conn:
        tags = json.loads(conn.execute(
            "SELECT m.dietary_tags FROM recipes r JOIN recipe_metadata m ON m.recipe_id = r.id WHERE r.title = ?",
            (SEED_RECIPES[0]["title"],),
        ).fetchone()[0])
    assert set(tags) >= {"vegetarian", "vegan"}
