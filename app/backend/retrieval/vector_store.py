from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Vector store using FAISS IndexFlatIP (Cosine Similarity) with NumPy fallback."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.recipe_ids: list[int] = []
        self._index: Any = None
        self._vectors: np.ndarray | None = None  # For NumPy fallback if needed
        self._use_faiss = True
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._use_faiss = True
        except ImportError:
            logger.warning("FAISS not installed; falling back to NumPy cosine similarity.")
            self._index = None
            self._use_faiss = False
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        return len(self.recipe_ids) == 0

    @property
    def total_vectors(self) -> int:
        return len(self.recipe_ids)

    def add(self, embeddings: np.ndarray, recipe_ids: list[int]) -> None:
        """Add normalized embeddings and their corresponding recipe IDs."""
        if embeddings.shape[0] != len(recipe_ids):
            raise ValueError("Number of embeddings does not match number of recipe IDs.")

        if embeddings.shape[0] == 0:
            return

        embeddings = embeddings.astype(np.float32)

        # Normalize in place if not already normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms

        if self._use_faiss and self._index is not None:
            self._index.add(normalized)
        else:
            if self._vectors is None or self._vectors.shape[0] == 0:
                self._vectors = normalized
            else:
                self._vectors = np.vstack([self._vectors, normalized])

        self.recipe_ids.extend(recipe_ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        min_threshold: float = 0.0,
    ) -> list[tuple[int, float, int]]:
        """Search top-K nearest recipes given a query vector.

        Returns:
            List of (recipe_id, similarity_score, rank_1_indexed) tuples.
        """
        if self.is_empty:
            return []

        # Prepare query vector
        query = query_vector.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        top_k = min(top_k, len(self.recipe_ids))
        results: list[tuple[int, float, int]] = []

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(query, top_k)
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0 or idx >= len(self.recipe_ids):
                    continue
                score_val = float(score)
                if score_val >= min_threshold:
                    results.append((self.recipe_ids[idx], score_val, rank))
        else:
            if self._vectors is None or len(self._vectors) == 0:
                return []
            # NumPy dot product for Cosine Similarity
            scores = np.dot(self._vectors, query.T).flatten()
            ranked_indices = np.argsort(-scores)[:top_k]
            for rank, idx in enumerate(ranked_indices, start=1):
                score_val = float(scores[idx])
                if score_val >= min_threshold:
                    results.append((self.recipe_ids[idx], score_val, rank))

        return results

    def save(self, index_path: Path, ids_path: Path) -> None:
        """Save FAISS index and recipe IDs to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(index_path))
        else:
            # Save numpy vectors
            if self._vectors is not None:
                np.save(str(index_path) + ".npy", self._vectors)

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "recipe_ids": self.recipe_ids,
                    "dimension": self.dimension,
                    "count": len(self.recipe_ids),
                },
                f,
                indent=2,
            )
        logger.info("Saved vector index to %s and IDs to %s", index_path, ids_path)

    def load(self, index_path: Path, ids_path: Path) -> bool:
        """Load index and recipe IDs from disk. Returns True if successfully loaded."""
        if not ids_path.exists():
            return False

        with open(ids_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.recipe_ids = [int(i) for i in data.get("recipe_ids", [])]
            self.dimension = int(data.get("dimension", self.dimension))

        if index_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(index_path))
                self._use_faiss = True
                return True
            except Exception as e:
                logger.warning("Failed to load FAISS index from %s: %s. Trying NumPy fallback.", index_path, e)

        numpy_path = Path(str(index_path) + ".npy")
        if numpy_path.exists():
            self._vectors = np.load(str(numpy_path))
            self._use_faiss = False
            return True

        return False
