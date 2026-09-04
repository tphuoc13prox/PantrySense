from __future__ import annotations

import logging
from typing import Any
import numpy as np

from app.backend.config import get_embedder_engine, get_embedding_model_name

logger = logging.getLogger(__name__)

_CACHED_CUDA_MODEL: Any = None
_CACHED_ONNX_MODEL: Any = None
_CACHED_CPU_MODEL: Any = None


class IngredientEmbedder:
    """High-speed dual-engine embedder supporting PyTorch CUDA GPU & FastEmbed ONNX INT8 CPU."""

    def __init__(self, model_name: str | None = None, engine: str | None = None) -> None:
        self.model_name = model_name or get_embedding_model_name()
        self.engine = (engine or get_embedder_engine()).strip().lower()
        self._dimension = 384
        self._cuda_model: Any = None
        self._onnx_model: Any = None
        self._cpu_model: Any = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_cuda_model(self) -> Any:
        global _CACHED_CUDA_MODEL
        if self._cuda_model is not None:
            return self._cuda_model
        if _CACHED_CUDA_MODEL is not None:
            self._cuda_model = _CACHED_CUDA_MODEL
            return self._cuda_model

        try:
            import torch
            from sentence_transformers import SentenceTransformer
            if torch.cuda.is_available():
                logger.info("Initializing PyTorch CUDA Embedding Model on GPU: %s", torch.cuda.get_device_name(0))
                model = SentenceTransformer(self.model_name, device="cuda")
                self._cuda_model = model
                _CACHED_CUDA_MODEL = model
                return self._cuda_model
        except Exception as e:
            logger.warning("Could not initialize CUDA model: %s. Falling back to ONNX/CPU.", e)
        return None

    def _get_onnx_model(self) -> Any:
        global _CACHED_ONNX_MODEL
        if self._onnx_model is not None:
            return self._onnx_model
        if _CACHED_ONNX_MODEL is not None:
            self._onnx_model = _CACHED_ONNX_MODEL
            return self._onnx_model

        try:
            from fastembed import TextEmbedding
            logger.info("Initializing FastEmbed ONNX INT8 Model for CPU acceleration...")
            model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self._onnx_model = model
            _CACHED_ONNX_MODEL = model
            return self._onnx_model
        except Exception as e:
            logger.warning("Could not initialize FastEmbed ONNX model: %s. Falling back to PyTorch CPU.", e)
        return None

    def _get_cpu_model(self) -> Any:
        global _CACHED_CPU_MODEL
        if self._cpu_model is not None:
            return self._cpu_model
        if _CACHED_CPU_MODEL is not None:
            self._cpu_model = _CACHED_CPU_MODEL
            return self._cpu_model

        from sentence_transformers import SentenceTransformer
        logger.info("Initializing standard PyTorch CPU SentenceTransformer...")
        model = SentenceTransformer(self.model_name, device="cpu")
        self._cpu_model = model
        _CACHED_CPU_MODEL = model
        return self._cpu_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings using the selected high-speed engine and return L2-normalized vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # 1. Try CUDA if engine is cuda
        if self.engine == "cuda":
            cuda_model = self._get_cuda_model()
            if cuda_model is not None:
                try:
                    embeddings = cuda_model.encode(
                        texts,
                        batch_size=256,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    return embeddings.astype(np.float32)
                except Exception as e:
                    logger.warning("CUDA embedding execution failed: %s, falling back to ONNX/CPU", e)

        # 2. Try FastEmbed ONNX INT8
        if self.engine in ("onnx", "fastembed", "cuda"):
            onnx_model = self._get_onnx_model()
            if onnx_model is not None:
                try:
                    raw_generator = onnx_model.embed(texts, batch_size=128)
                    embeddings_list = list(raw_generator)
                    arr = np.array(embeddings_list, dtype=np.float32)
                    # L2-normalize
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    return (arr / norms).astype(np.float32)
                except Exception as e:
                    logger.warning("FastEmbed ONNX execution failed: %s, falling back to standard CPU", e)

        # 3. Fallback to standard PyTorch CPU
        cpu_model = self._get_cpu_model()
        embeddings = cpu_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and return a 1D float32 normalized vector."""
        if not text:
            return np.zeros((self.dimension,), dtype=np.float32)
        embeddings = self.embed_texts([text])
        return embeddings[0]
