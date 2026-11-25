from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.backend.config import get_embedder_engine, get_embedding_model_name

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_METADATA_SUFFIX = ".meta.json"
INDEX_METADATA_VERSION = 1

BACKEND_SENTENCE_TRANSFORMERS = "sentence-transformers"
BACKEND_FASTEMBED = "fastembed"

# Checkpoints whose FastEmbed ONNX export gives the same vectors as
# sentence-transformers (cosine 1.000000 on 400 queries and 400 recipe texts,
# see docs/embedding.md). Only these may use FastEmbed; everything else runs
# through sentence-transformers so the query side always matches the index.
FASTEMBED_EQUIVALENT_MODELS = frozenset({DEFAULT_EMBEDDING_MODEL})

# Asymmetric models that expect a role prefix on each input.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
}

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class EmbeddingIndexMismatchError(RuntimeError):
    """Raised when a vector index was built with a different embedding setup than the query side."""


def resolve_prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) for a model name or local model directory.

    Local sentence-transformers directories can declare prompts in
    config_sentence_transformers.json; those take priority over the built-in table.
    """
    config_path = Path(model_name) / "config_sentence_transformers.json"
    try:
        if config_path.is_file():
            prompts = json.loads(config_path.read_text(encoding="utf-8")).get("prompts") or {}
            if prompts:
                query = prompts.get("query", "")
                passage = prompts.get("passage", prompts.get("document", ""))
                return query, passage
    except (OSError, ValueError) as exc:
        logger.warning("Could not read prompts from %s: %s", config_path, exc)

    if model_name in KNOWN_PREFIXES:
        return KNOWN_PREFIXES[model_name]
    if model_name.startswith("intfloat/e5-") or model_name.startswith("intfloat/multilingual-e5-"):
        return "query: ", "passage: "
    return "", ""


def _normalize_model_id(model_name: str) -> str:
    """Make local paths comparable regardless of how they were spelled."""
    path = Path(model_name)
    if path.exists():
        return str(path.resolve()).replace("\\", "/").lower()
    return model_name.strip()


def same_embedding_model(index_model: str, query_model: str) -> bool:
    if _normalize_model_id(index_model) == _normalize_model_id(query_model):
        return True
    # An index records the absolute path of a local model. A copy of that folder
    # elsewhere (e.g. in another checkout's data/models/) is accepted when the
    # folder name matches; prefixes and dimension are still checked.
    query_path = Path(query_model)
    index_name = index_model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    is_local_index_model = "/" in index_model.replace("\\", "/")
    return is_local_index_model and query_path.is_dir() and index_name.lower() == query_path.name.lower()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_cached(key: tuple[str, str, str], factory: Any) -> Any:
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = factory()
            _MODEL_CACHE[key] = model
        return model


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class IngredientEmbedder:
    """Sentence embedder for recipe passages and ingredient queries.

    `engine` is a preference ("cuda", "onnx"/"fastembed", "cpu"). The backend that
    actually runs is recorded in `backend` so an index can be checked against the
    embedder that will query it.
    """

    def __init__(self, model_name: str | None = None, engine: str | None = None) -> None:
        self.model_name = (model_name or get_embedding_model_name()).strip()
        self.engine = (engine or get_embedder_engine()).strip().lower()
        self.query_prefix, self.passage_prefix = resolve_prefixes(self.model_name)
        self._dimension = 384
        self._dimension_known = False
        self._backend: str | None = None
        self._device: str | None = None
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._backend, self._device = self._choose_backend()
        return self._backend

    def _choose_backend(self) -> tuple[str, str]:
        if self.engine == "cuda" and _cuda_available():
            return BACKEND_SENTENCE_TRANSFORMERS, "cuda"
        if self.engine in ("onnx", "fastembed", "cuda") and self.model_name in FASTEMBED_EQUIVALENT_MODELS:
            return BACKEND_FASTEMBED, "cpu"
        if self.engine in ("onnx", "fastembed"):
            logger.info(
                "FastEmbed has no verified export of '%s'; using sentence-transformers on CPU.",
                self.model_name,
            )
        return BACKEND_SENTENCE_TRANSFORMERS, "cpu"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        backend = self.backend
        device = self._device or "cpu"
        if backend == BACKEND_FASTEMBED:
            try:
                from fastembed import TextEmbedding

                model = _load_cached(
                    (backend, self.model_name, device),
                    lambda: TextEmbedding(model_name=self.model_name),
                )
                self._model = model
                return model
            except Exception as exc:
                logger.warning("FastEmbed unavailable (%s); using sentence-transformers on CPU.", exc)
                self._backend, self._device = BACKEND_SENTENCE_TRANSFORMERS, "cpu"
                backend, device = self._backend, self._device

        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model '%s' on %s", self.model_name, device)
        model = _load_cached(
            (backend, self.model_name, device),
            lambda: SentenceTransformer(self.model_name, device=device),
        )
        get_dim = getattr(model, "get_sentence_embedding_dimension", None) or getattr(
            model, "get_embedding_dimension", None
        )
        dim = get_dim() if get_dim else None
        if dim:
            self._dimension = int(dim)
            self._dimension_known = True
        self._model = model
        return model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        if self.backend == BACKEND_FASTEMBED:
            arr = _l2_normalize(np.array(list(model.embed(texts, batch_size=128)), dtype=np.float32))
            self._dimension = int(arr.shape[1])
            self._dimension_known = True
            return arr
        embeddings = model.encode(
            texts,
            batch_size=256 if self._device == "cuda" else 64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._encode([self.query_prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed recipe passages (the index side). Returns L2-normalized float32 vectors."""
        return self.embed_passages(texts)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one ingredient query. Returns a 1D L2-normalized float32 vector."""
        if not text:
            return np.zeros((self.dimension,), dtype=np.float32)
        return self.embed_queries([text])[0]

    def index_metadata(self) -> dict[str, Any]:
        """Describe the embedding setup so it can be stored next to an index."""
        return {
            "format_version": INDEX_METADATA_VERSION,
            "model": str(Path(self.model_name).resolve()) if Path(self.model_name).exists() else self.model_name,
            "backend": self.backend,
            "dimension": self.dimension,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "normalized": True,
        }

    def verify_index(self, index_path: Path) -> None:
        """Raise EmbeddingIndexMismatchError if the index at `index_path` was built differently."""
        metadata = read_index_metadata(index_path)
        if metadata is None:
            model = getattr(self, "model_name", DEFAULT_EMBEDDING_MODEL)
            if model != DEFAULT_EMBEDDING_MODEL:
                logger.warning(
                    "Index %s has no embedding metadata and is assumed to use %s, but queries use %s. "
                    "Rebuild the index with this model.",
                    index_path,
                    DEFAULT_EMBEDDING_MODEL,
                    model,
                )
            return
        check_index_compatibility(metadata, self)


def index_metadata_path(index_path: Path) -> Path:
    return Path(str(index_path) + INDEX_METADATA_SUFFIX)


def write_index_metadata(index_path: Path, embedder: IngredientEmbedder, **extra: Any) -> Path:
    path = index_metadata_path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**embedder.index_metadata(), **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_index_metadata(index_path: Path) -> dict[str, Any] | None:
    path = index_metadata_path(index_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_index_compatibility(metadata: dict[str, Any], embedder: IngredientEmbedder) -> None:
    """Compare stored index metadata with the embedder that will produce query vectors."""
    problems: list[str] = []
    index_model = str(metadata.get("model", ""))
    query_model = str(getattr(embedder, "model_name", ""))
    if not same_embedding_model(index_model, query_model):
        problems.append(f"model {index_model!r} != {query_model!r}")

    query_prefix = getattr(embedder, "query_prefix", "")
    passage_prefix = getattr(embedder, "passage_prefix", "")
    if metadata.get("query_prefix", "") != query_prefix or metadata.get("passage_prefix", "") != passage_prefix:
        problems.append("query/passage prefixes differ")

    if getattr(embedder, "_dimension_known", False) and int(metadata.get("dimension", 0)) != embedder.dimension:
        problems.append(f"dimension {metadata.get('dimension')} != {embedder.dimension}")

    index_backend = metadata.get("backend")
    query_backend = embedder.backend if hasattr(embedder, "engine") else index_backend
    if index_backend != query_backend and query_model not in FASTEMBED_EQUIVALENT_MODELS:
        problems.append(f"backend {index_backend!r} != {query_backend!r}")

    if problems:
        raise EmbeddingIndexMismatchError(
            "Vector index does not match the query embedder: "
            + "; ".join(problems)
            + ". Rebuild the index or set PANTRYSENSE_EMBEDDING_MODEL to the index model."
        )
