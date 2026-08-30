from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "data" / "recipes.db"
DEFAULT_VECTOR_INDEX_PATH = BASE_DIR / "data" / "recipe_vectors.index"
DEFAULT_VECTOR_IDS_PATH = BASE_DIR / "data" / "recipe_vector_ids.json"


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
    return os.getenv("PANTRYSENSE_RETRIEVAL_MODE", "semantic").strip().lower()


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
