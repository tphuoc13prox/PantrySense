from __future__ import annotations

from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "data" / "recipes.db"


def get_database_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_DB_PATH", DEFAULT_DB_PATH))


def get_match_threshold() -> float:
    return float(
        os.getenv(
            "PANTRYSENSE_MINIMUM_COVERAGE",
            os.getenv("PANTRYSENSE_MATCH_THRESHOLD", "0.34"),
        )
    )
