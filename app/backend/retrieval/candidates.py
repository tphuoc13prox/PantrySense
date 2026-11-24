from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMatch:
    """A recipe returned by a retriever, with whichever scores that retriever produced."""

    recipe_id: int
    semantic_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_rank: int = 1
