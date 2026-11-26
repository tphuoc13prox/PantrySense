"""ingest_dataset: id numbering and seed recipes match the published corpus layout."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.backend.database.seed_data import SEED_RECIPES
from app.backend.system import setup_pipeline
from app.backend.system.core_catalog import CORE_RECIPE_CATALOG


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No dataset download: the ingest falls back to the built-in catalog."""
    def no_download(progress_callback=None):
        raise OSError("offline")

    monkeypatch.setattr(setup_pipeline, "download_full_hf_parquet_file", no_download)
    monkeypatch.setattr(setup_pipeline, "fetch_online_recipes_from_huggingface", lambda limit: [])


def _rows(db: Path) -> dict[str, tuple[int, list[str]]]:
    with sqlite3.connect(db) as conn:
        return {t: (i, json.loads(ings)) for i, t, ings in conn.execute("SELECT id, title, ingredients FROM recipes")}


def test_fresh_database_starts_at_the_published_first_id(tmp_path: Path, offline: None) -> None:
    db = tmp_path / "recipes.db"
    total = setup_pipeline.ingest_dataset(db, 100)

    rows = _rows(db)
    catalog = {r["title"] for r in CORE_RECIPE_CATALOG}
    assert total == len(catalog)
    # Seeds without a dataset row of the same title are added after it.
    assert len(rows) == len(catalog | {s["title"] for s in SEED_RECIPES})
    first = setup_pipeline.FIRST_RECIPE_ID
    assert sorted(rows[t][0] for t in catalog) == list(range(first, first + len(catalog)))
    # Seed recipes are applied over catalog rows with the same title, as on app start.
    for seed in SEED_RECIPES:
        if seed["title"] in rows:
            assert rows[seed["title"]][1] == seed["ingredients"]


def test_ids_of_an_earlier_dataset_are_not_reused(tmp_path: Path, offline: None) -> None:
    db = tmp_path / "recipes.db"
    setup_pipeline.ingest_dataset(db, 100)
    first_max = max(i for i, _ in _rows(db).values())

    setup_pipeline.ingest_dataset(db, 100)
    assert min(i for i, _ in _rows(db).values()) > first_max


def test_ingest_drops_the_old_vocabulary_cache(tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = tmp_path / "ingredient_vocab.json"
    vocab.write_text(json.dumps({"ingredient from the old dataset": 5}), encoding="utf-8")
    monkeypatch.setenv("PANTRYSENSE_INGREDIENT_VOCAB_PATH", str(vocab))
    setup_pipeline.ingest_dataset(tmp_path / "recipes.db", 100)
    assert not vocab.exists()
