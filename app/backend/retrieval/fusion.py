from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMatch:
    recipe_id: int
    semantic_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_rank: int = 1


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float, int]],
    sparse_results: list[tuple[int, float, int]],
    k: int = 60,
    top_k: int = 20,
) -> list[CandidateMatch]:
    """Fuse dense (semantic) and sparse (lexical/BM25) rankings using Reciprocal Rank Fusion.

    Args:
        dense_results: List of (recipe_id, semantic_score, rank_1_indexed).
        sparse_results: List of (recipe_id, bm25_score, rank_1_indexed).
        k: Smoothing constant for RRF (default: 60).
        top_k: Number of fused candidates to return.

    Returns:
        List of CandidateMatch sorted descending by RRF score.
    """
    scores: dict[int, float] = {}
    dense_map: dict[int, tuple[float, int]] = {}
    sparse_map: dict[int, tuple[float, int]] = {}

    for recipe_id, score, rank in dense_results:
        dense_map[recipe_id] = (score, rank)
        scores[recipe_id] = scores.get(recipe_id, 0.0) + (1.0 / (k + rank))

    for recipe_id, score, rank in sparse_results:
        sparse_map[recipe_id] = (score, rank)
        scores[recipe_id] = scores.get(recipe_id, 0.0) + (1.0 / (k + rank))

    if not scores:
        return []

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(scores.items(), key=lambda item: -item[1])[:top_k]

    fused_matches: list[CandidateMatch] = []
    for final_rank, (recipe_id, rrf_score) in enumerate(sorted_candidates, start=1):
        sem_info = dense_map.get(recipe_id)
        bm25_info = sparse_map.get(recipe_id)

        fused_matches.append(
            CandidateMatch(
                recipe_id=recipe_id,
                semantic_score=round(sem_info[0], 4) if sem_info else None,
                bm25_score=round(bm25_info[0], 4) if bm25_info else None,
                rrf_score=round(rrf_score, 6),
                retrieval_rank=final_rank,
            )
        )

    return fused_matches
