from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "data" / "recipes.db"
DEFAULT_VECTOR_INDEX_PATH = BASE_DIR / "data" / "recipe_vectors.index"
DEFAULT_VECTOR_IDS_PATH = BASE_DIR / "data" / "recipe_vector_ids.json"
DEFAULT_BM25_INDEX_PATH = BASE_DIR / "data" / "recipe_bm25.json"


def get_database_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_DB_PATH", DEFAULT_DB_PATH))


def get_match_threshold() -> float:
    return float(
        os.getenv(
            "PANTRYSENSE_MINIMUM_COVERAGE",
            os.getenv("PANTRYSENSE_MATCH_THRESHOLD", "0.34"),
        )
    )


def get_retrieval_mode() -> str:
    return os.getenv("PANTRYSENSE_RETRIEVAL_MODE", "hybrid").strip().lower()


def get_embedding_model_name() -> str:
    return os.getenv("PANTRYSENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()


def get_semantic_top_k() -> int:
    return int(os.getenv("PANTRYSENSE_SEMANTIC_TOP_K", "20"))


def get_semantic_threshold() -> float:
    return float(os.getenv("PANTRYSENSE_SEMANTIC_THRESHOLD", "0.35"))


def get_vector_index_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_VECTOR_INDEX_PATH", DEFAULT_VECTOR_INDEX_PATH))


def get_vector_ids_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_VECTOR_IDS_PATH", DEFAULT_VECTOR_IDS_PATH))


def get_bm25_index_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_BM25_INDEX_PATH", DEFAULT_BM25_INDEX_PATH))


def get_rrf_k() -> int:
    return int(os.getenv("PANTRYSENSE_RRF_K", "60"))


def get_auto_open_browser() -> bool:
    val = os.getenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "true").strip().lower()
    return val in ("true", "1", "yes", "on")


def get_auto_shutdown_enabled() -> bool:
    val = os.getenv("PANTRYSENSE_AUTO_SHUTDOWN", "true").strip().lower()
    return val in ("true", "1", "yes", "on")


def get_inactive_timeout_seconds() -> float:
    return float(os.getenv("PANTRYSENSE_INACTIVE_TIMEOUT", "10.0"))


def get_startup_grace_period_seconds() -> float:
    return float(os.getenv("PANTRYSENSE_STARTUP_GRACE_PERIOD", "15.0"))
