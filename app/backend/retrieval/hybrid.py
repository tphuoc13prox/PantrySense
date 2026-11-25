from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.backend.config import (
    get_bm25_index_path,
    get_rrf_k,
    get_semantic_threshold,
    get_semantic_top_k,
)
from app.backend.retrieval.bm25_store import BM25Store, preferred_index_path
from app.backend.retrieval.candidates import CandidateMatch
from app.backend.retrieval.fusion import reciprocal_rank_fusion
from app.backend.retrieval.semantic import (
    SemanticIndexNotFoundError,
    SemanticRetriever,
    build_query_representation,
)

if TYPE_CHECKING:
    from app.backend.retrieval.filtering import AllowedIds

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
        self._bm25_path = bm25_path
        self._loaded = False
        self._bm25_lock = threading.Lock()

    @property
    def bm25_path(self) -> Path:
        return self._bm25_path if self._bm25_path is not None else get_bm25_index_path()


    def ensure_bm25_loaded(self) -> bool:
        """Load the BM25 index once; prefers the ``.npz`` sibling of a JSON path."""
        if not self.bm25_store.is_empty:
            return True
        with self._bm25_lock:
            if not self.bm25_store.is_empty:
                return True
            path = preferred_index_path(self.bm25_path)
            if not path.exists():
                return False
            return self.bm25_store.load(path)

    def ensure_loaded(self) -> bool:
        """Attempt to load both semantic and BM25 indices from disk."""
        sem_ready = self.semantic_retriever.ensure_loaded()
        bm25_ready = self.ensure_bm25_loaded()
        self._loaded = sem_ready and bm25_ready
        return self._loaded

    def is_ready(self) -> bool:
        return self.ensure_loaded() and not self.bm25_store.is_empty and self.semantic_retriever.is_ready()

    def retrieve(
        self,
        ingredients: list[str],
        top_k: int | None = None,
        k_rrf: int | None = None,
        allowed: "AllowedIds | None" = None,
    ) -> list[CandidateMatch]:
        """Execute hybrid search using both BM25 and FAISS, then fuse with RRF.

        With ``allowed`` both retrievers search only inside that id set, so
        filtering happens before the top-k cut instead of after it.
        """
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
        if allowed is not None and allowed.is_empty:
            return []

        # 1. Dense Semantic Search
        dense_matches = self.semantic_retriever.vector_store.search(
            self.semantic_retriever.embedder.embed_text(query_str),
            top_k=k_limit * 2,
            min_threshold=get_semantic_threshold(),
            allowed=allowed,
        )

        # 2. Sparse Lexical BM25 Search
        sparse_matches = self.bm25_store.search(
            query=query_str,
            top_k=k_limit * 2,
            allowed=allowed,
        )

        # 3. Reciprocal Rank Fusion
        return reciprocal_rank_fusion(
            dense_results=dense_matches,
            sparse_results=sparse_matches,
            k=rrf_k_param,
            top_k=k_limit,
        )
