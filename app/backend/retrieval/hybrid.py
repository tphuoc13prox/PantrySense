from __future__ import annotations

import logging
from pathlib import Path

from app.backend.config import (
    get_bm25_index_path,
    get_rrf_k,
    get_semantic_threshold,
    get_semantic_top_k,
    get_vector_ids_path,
    get_vector_index_path,
)
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.fusion import CandidateMatch, reciprocal_rank_fusion
from app.backend.retrieval.semantic import (
    SemanticIndexNotFoundError,
    SemanticRetriever,
    build_query_representation,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 lexical retrieval and FAISS semantic retrieval using RRF."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever | None = None,
        bm25_store: BM25Store | None = None,
        bm25_path: Path | None = None,
    ) -> None:
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.bm25_store = bm25_store or BM25Store()
        self.bm25_path = bm25_path or get_bm25_index_path()
        self._loaded = False

    def ensure_loaded(self) -> bool:
        """Attempt to load both semantic and BM25 indices from disk."""
        sem_ready = self.semantic_retriever.ensure_loaded()

        if not self.bm25_store.is_empty:
            bm25_ready = True
        elif self.bm25_path.exists():
            bm25_ready = self.bm25_store.load(self.bm25_path)
        else:
            bm25_ready = False

        self._loaded = sem_ready and bm25_ready
        return self._loaded

    def is_ready(self) -> bool:
        return self.ensure_loaded() and not self.bm25_store.is_empty and self.semantic_retriever.is_ready()

    def retrieve(
        self,
        ingredients: list[str],
        top_k: int | None = None,
        k_rrf: int | None = None,
    ) -> list[CandidateMatch]:
        """Execute hybrid search using both BM25 and FAISS, then fuse with RRF."""
        if not ingredients:
            return []

        if not self.is_ready():
            raise SemanticIndexNotFoundError(
                "Hybrid index (FAISS or BM25) not found. "
                "Please run 'python scripts/build_vector_index.py' to generate indices."
            )

        k_limit = top_k if top_k is not None else get_semantic_top_k()
        rrf_k_param = k_rrf if k_rrf is not None else get_rrf_k()

        query_str = build_query_representation(ingredients)
        if not query_str:
            return []

        # 1. Dense Semantic Search
        dense_matches = self.semantic_retriever.vector_store.search(
            self.semantic_retriever.embedder.embed_text(query_str),
            top_k=k_limit * 2,
            min_threshold=get_semantic_threshold(),
        )

        # 2. Sparse Lexical BM25 Search
        sparse_matches = self.bm25_store.search(
            query=query_str,
            top_k=k_limit * 2,
        )

        # 3. Reciprocal Rank Fusion
        return reciprocal_rank_fusion(
            dense_results=dense_matches,
            sparse_results=sparse_matches,
            k=rrf_k_param,
            top_k=k_limit,
        )
