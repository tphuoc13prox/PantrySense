from __future__ import annotations

from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.semantic import (
    CandidateMatch,
    SemanticIndexNotFoundError,
    SemanticRetriever,
    build_query_representation,
    build_recipe_representation,
)
from app.backend.retrieval.vector_store import FaissVectorStore

__all__ = [
    "CandidateMatch",
    "FaissVectorStore",
    "IngredientEmbedder",
    "SemanticIndexNotFoundError",
    "SemanticRetriever",
    "build_query_representation",
    "build_recipe_representation",
]
