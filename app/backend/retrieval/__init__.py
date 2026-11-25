from __future__ import annotations

from app.backend.retrieval.bm25_store import BM25Store, tokenize
from app.backend.retrieval.candidates import CandidateMatch
from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.filtering import AllowedIds
from app.backend.retrieval.fusion import reciprocal_rank_fusion
from app.backend.retrieval.hybrid import HybridRetriever
from app.backend.retrieval.semantic import (
    SemanticIndexNotFoundError,
    SemanticRetriever,
    build_query_representation,
    build_recipe_representation,
)
from app.backend.retrieval.vector_store import FaissVectorStore

__all__ = [
    "AllowedIds",
    "BM25Store",
    "CandidateMatch",
    "FaissVectorStore",
    "HybridRetriever",
    "IngredientEmbedder",
    "SemanticIndexNotFoundError",
    "SemanticRetriever",
    "build_query_representation",
    "build_recipe_representation",
    "reciprocal_rank_fusion",
    "tokenize",
]
