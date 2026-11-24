from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.backend.config import (
    get_semantic_threshold,
    get_semantic_top_k,
    get_vector_ids_path,
    get_vector_index_path,
)
from app.backend.recipes.matching import normalize_ingredients
from app.backend.retrieval.candidates import CandidateMatch
from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.vector_store import FaissVectorStore

if TYPE_CHECKING:
    from app.backend.retrieval.filtering import AllowedIds

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateMatch",
    "SemanticIndexNotFoundError",
    "SemanticRetriever",
    "build_query_representation",
    "build_recipe_representation",
]


class SemanticIndexNotFoundError(RuntimeError):
    """Raised when the FAISS semantic vector index is missing."""
    pass


def build_query_representation(ingredients: list[str]) -> str:
    """Build a deterministic query string from user-supplied ingredients."""
    normalized = normalize_ingredients(ingredients)
    return " ".join(normalized)


def build_recipe_representation(title: str, ingredients: list[str]) -> str:
    """Build a deterministic text representation of a recipe for embedding.
    Format: Recipe Title + newline + normalized ingredient list
    """
    normalized = normalize_ingredients(ingredients)
    return f"{title.strip()}\n{' '.join(normalized)}"


class SemanticRetriever:
    """Retrieves top-K candidate recipes based on semantic similarity."""

    def __init__(
        self,
        embedder: IngredientEmbedder | None = None,
        vector_store: FaissVectorStore | None = None,
        index_path: Path | None = None,
        ids_path: Path | None = None,
    ) -> None:
        self.embedder = embedder or IngredientEmbedder()
        self.vector_store = vector_store or FaissVectorStore(dimension=self.embedder.dimension)
        self._index_path = index_path
        self._ids_path = ids_path
        self._loaded = False
        self._load_lock = threading.Lock()

    @property
    def index_path(self) -> Path:
        return self._index_path if self._index_path is not None else get_vector_index_path()

    @property
    def ids_path(self) -> Path:
        return self._ids_path if self._ids_path is not None else get_vector_ids_path()


    def ensure_loaded(self) -> bool:
        """Attempt to load the index from disk if not already loaded."""
        if self._loaded and not self.vector_store.is_empty:
            return True

        with self._load_lock:
            if self._loaded and not self.vector_store.is_empty:
                return True
            if self.ids_path.exists():
                success = self.vector_store.load(self.index_path, self.ids_path)
                self._loaded = success
                return success

        return False

    def is_ready(self) -> bool:
        """Check if vector store is loaded and ready for search."""
        return self.ensure_loaded() and not self.vector_store.is_empty

    def retrieve(
        self,
        ingredients: list[str],
        top_k: int | None = None,
        min_threshold: float | None = None,
        allowed: "AllowedIds | None" = None,
    ) -> list[CandidateMatch]:
        """Retrieve candidate recipe IDs matching user ingredients semantically.

        ``allowed`` limits the search to recipes that passed metadata filters.
        """
        if not ingredients:
            return []
        if allowed is not None and allowed.is_empty:
            return []

        if not self.is_ready():
            raise SemanticIndexNotFoundError(
                f"Semantic vector index not found at '{self.index_path}'. "
                "Please run 'python scripts/build_vector_index.py' to generate the index."
            )

        k = top_k if top_k is not None else get_semantic_top_k()
        threshold = min_threshold if min_threshold is not None else get_semantic_threshold()

        query_str = build_query_representation(ingredients)
        if not query_str:
            return []

        query_vector = self.embedder.embed_text(query_str)
        search_results = self.vector_store.search(
            query_vector,
            top_k=k,
            min_threshold=threshold,
            allowed=allowed,
        )

        return [
            CandidateMatch(
                recipe_id=recipe_id,
                semantic_score=round(score, 4),
                retrieval_rank=rank,
            )
            for recipe_id, score, rank in search_results
        ]
