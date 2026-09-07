from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "data" / "recipes.db"
DEFAULT_VECTOR_INDEX_PATH = BASE_DIR / "data" / "recipe_vectors.index"
DEFAULT_VECTOR_IDS_PATH = BASE_DIR / "data" / "recipe_vector_ids.json"
DEFAULT_BM25_INDEX_PATH = BASE_DIR / "data" / "recipe_bm25.json"
DEFAULT_RANKER_MODEL_PATH = BASE_DIR / "data" / "ranker_model.joblib"


def get_database_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_DB_PATH", DEFAULT_DB_PATH))


def get_match_threshold() -> float:
    return float(
        os.getenv(
            "PANTRYSENSE_MINIMUM_COVERAGE",
            os.getenv("PANTRYSENSE_MATCH_THRESHOLD", "0.28"),
        )
    )



def get_retrieval_mode() -> str:
    return os.getenv("PANTRYSENSE_RETRIEVAL_MODE", "hybrid").strip().lower()


def get_ranking_mode() -> str:
    return os.getenv("PANTRYSENSE_RANKING_MODE", "ml").strip().lower()


def get_ranker_model_path() -> Path:
    return Path(os.getenv("PANTRYSENSE_RANKER_MODEL_PATH", DEFAULT_RANKER_MODEL_PATH))


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


def get_auto_build_index() -> bool:
    val = os.getenv("PANTRYSENSE_AUTO_BUILD_INDEX", "true").strip().lower()
    return val in ("true", "1", "yes", "on")


def get_tab_ttl_seconds() -> float:
    return float(os.getenv("PANTRYSENSE_TAB_TTL", "60.0"))


def get_inactive_timeout_seconds() -> float:
    return float(os.getenv("PANTRYSENSE_INACTIVE_TIMEOUT", "15.0"))


_CURRENT_EMBEDDER_ENGINE: str | None = None


def get_embedder_engine() -> str:
    global _CURRENT_EMBEDDER_ENGINE
    if _CURRENT_EMBEDDER_ENGINE:
        return _CURRENT_EMBEDDER_ENGINE
    env_engine = os.getenv("PANTRYSENSE_EMBEDDER_ENGINE", "").strip().lower()
    if env_engine in ("cuda", "onnx", "fastembed", "cpu"):
        return env_engine
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "onnx"


def set_embedder_engine(engine: str) -> None:
    global _CURRENT_EMBEDDER_ENGINE
    clean = engine.strip().lower()
    if clean in ("cuda", "onnx", "fastembed", "cpu"):
        _CURRENT_EMBEDDER_ENGINE = clean


def get_dataset_download_limit() -> int:
    return int(os.getenv("PANTRYSENSE_DATASET_LIMIT", "380000"))


def get_startup_grace_period_seconds() -> float:
    return float(os.getenv("PANTRYSENSE_STARTUP_GRACE_PERIOD", "30.0"))

