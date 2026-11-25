"""Ranking metrics over graded labels listed in ranked order."""
from __future__ import annotations

import math
from collections.abc import Sequence


def dcg_at_k(labels: Sequence[float], k: int) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(i + 1.0) for i, rel in enumerate(labels[:k], start=1))


def ndcg_at_k(ranked_labels: Sequence[float], k: int, all_labels: Sequence[float] | None = None) -> float:
    """nDCG@k. ``all_labels`` sets the ideal ordering (defaults to ranked_labels)."""
    ideal = dcg_at_k(sorted(all_labels if all_labels is not None else ranked_labels, reverse=True), k)
    return dcg_at_k(ranked_labels, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_labels: Sequence[float], k: int) -> float:
    for i, rel in enumerate(ranked_labels[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked_labels: Sequence[float], k: int) -> float:
    return 1.0 if any(rel > 0 for rel in ranked_labels[:k]) else 0.0


def summarize(per_query: Sequence[Sequence[float]], k: int = 10) -> dict[str, float]:
    """Mean nDCG@k, MRR@k, Hit@1 and Hit@k over queries (each a ranked label list)."""
    n = len(per_query)
    if n == 0:
        return {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0, "hit@1": 0.0, f"hit@{k}": 0.0, "queries": 0}
    return {
        f"ndcg@{k}": sum(ndcg_at_k(q, k) for q in per_query) / n,
        f"mrr@{k}": sum(mrr_at_k(q, k) for q in per_query) / n,
        "hit@1": sum(hit_at_k(q, 1) for q in per_query) / n,
        f"hit@{k}": sum(hit_at_k(q, k) for q in per_query) / n,
        "queries": n,
    }
