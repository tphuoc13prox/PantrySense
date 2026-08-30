from __future__ import annotations

import logging
from typing import Any
import numpy as np

from app.backend.config import get_embedding_model_name

logger = logging.getLogger(__name__)


class IngredientEmbedder:
    """Embedder for recipes and ingredient queries using SentenceTransformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_embedding_model_name()
        self._model: Any = None
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info("Loading sentence transformer model: %s", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device="cpu")
                # Update dimension dynamically if available
                dim = self._model.get_sentence_embedding_dimension()
                if dim is not None:
                    self._dimension = int(dim)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer model %s: %s", self.model_name, e)
                raise RuntimeError(
                    f"Could not load embedding model '{self.model_name}'. "
                    f"Ensure 'sentence-transformers' is installed. Error: {e}"
                ) from e
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings and return L2-normalized vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and return a 1D float32 normalized vector."""
        if not text:
            vector = np.zeros((self.dimension,), dtype=np.float32)
            return vector
        embeddings = self.embed_texts([text])
        return embeddings[0]
