"""Shared pytest setup.

Tests marked ``requires_data`` or ``requires_models`` are skipped when the
corresponding files under ``data/`` do not exist, so the suite runs on a bare
checkout (for example in CI) without the dataset or any built artifacts.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Some tests create a tiny seed recipes.db at the default path, so the file
# alone does not prove the real dataset is present; require a row count too.
MIN_REAL_DATASET_ROWS = 10_000

REQUIRED_FILES = {
    "requires_data": ("recipes.db",),
    "requires_models": (
        "recipe_vectors.index",
        "recipe_vector_ids.json",
        "recipe_bm25.json",
        "ranker_model.joblib",
    ),
}

# Keep tests from opening a browser or arming the idle shutdown timer.
os.environ.setdefault("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
os.environ.setdefault("PANTRYSENSE_AUTO_SHUTDOWN", "false")


def _recipe_count(db_path: Path) -> int:
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _missing(marker: str) -> list[str]:
    missing = [name for name in REQUIRED_FILES[marker] if not (DATA_DIR / name).exists()]
    if marker == "requires_data" and not missing and _recipe_count(DATA_DIR / "recipes.db") < MIN_REAL_DATASET_ROWS:
        missing = [f"recipes.db with at least {MIN_REAL_DATASET_ROWS} recipes"]
    return missing


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cache: dict[str, list[str]] = {}
    for item in items:
        for marker in REQUIRED_FILES:
            if item.get_closest_marker(marker) is None:
                continue
            if marker not in cache:
                cache[marker] = _missing(marker)
            if cache[marker]:
                reason = f"{marker}: missing {', '.join(cache[marker])} in data/"
                item.add_marker(pytest.mark.skip(reason=reason))
