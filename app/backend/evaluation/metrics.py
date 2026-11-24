"""Ranking metrics and bootstrap confidence intervals for offline evaluation.

All metric functions take a ranked list of document ids (best first) and the set
of relevant ids for one query, and return a float in [0, 1]. They are pure and
dependency-free so they can be reused for LTR label generation or any other
offline experiment.

Conventions:
- Duplicate ids in ``ranked`` are ignored after their first occurrence.
- An empty relevant set yields 0.0 for every metric (the query cannot be
  satisfied); callers should normally drop such queries before aggregating.
- ``k`` must be a positive integer.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

__all__ = [
    "dedupe",
    "first_relevant_rank",
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mrr_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "percentile",
    "bootstrap_ci",
    "paired_bootstrap_diff",
]


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def dedupe(ranked: Iterable[Hashable]) -> list[Hashable]:
    """Return ``ranked`` with later duplicates removed, order preserved."""
    seen: set[Hashable] = set()
    out: list[Hashable] = []
    for doc in ranked:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def first_relevant_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> int | None:
    """1-based rank of the first relevant id in ``ranked``, or None if absent."""
    rel = set(relevant)
    for rank, doc in enumerate(dedupe(ranked), start=1):
        if doc in rel:
            return rank
    return None


def hit_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """1.0 if at least one relevant id appears in the top ``k``, else 0.0."""
    _check_k(k)
    rank = first_relevant_rank(ranked, relevant)
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Fraction of relevant ids found in the top ``k``."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(dedupe(ranked)[:k])
    return len(top & rel) / len(rel)


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Fraction of the top ``k`` slots holding a relevant id (denominator is ``k``)."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return 0.0
    top = dedupe(ranked)[:k]
    return sum(1 for doc in top if doc in rel) / k


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """1 / rank of the first relevant id (no cutoff), 0.0 if none is retrieved."""
    rank = first_relevant_rank(ranked, relevant)
    return 1.0 / rank if rank is not None else 0.0


def mrr_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Reciprocal rank truncated at ``k`` (0.0 if the first relevant id is below ``k``)."""
    _check_k(k)
    rank = first_relevant_rank(ranked, relevant)
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def _gains(relevant: Iterable[Hashable] | Mapping[Hashable, float]) -> dict[Hashable, float]:
    if isinstance(relevant, Mapping):
        return {doc: float(g) for doc, g in relevant.items() if float(g) > 0}
    return {doc: 1.0 for doc in relevant}


def dcg_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable] | Mapping[Hashable, float], k: int) -> float:
    """Discounted cumulative gain with linear gains: sum(gain_i / log2(i + 1))."""
    _check_k(k)
    gains = _gains(relevant)
    return sum(
        gains.get(doc, 0.0) / math.log2(rank + 1)
        for rank, doc in enumerate(dedupe(ranked)[:k], start=1)
    )


def ndcg_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable] | Mapping[Hashable, float], k: int) -> float:
    """Normalised DCG. ``relevant`` is a set (binary gains) or a {id: gain} mapping."""
    _check_k(k)
    gains = _gains(relevant)
    if not gains:
        return 0.0
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg_at_k(ranked, gains, k) / idcg if idcg > 0 else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    data = sorted(values)
    pos = (len(data) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over queries.

    Returns ``(point_estimate, lower, upper)`` where the point estimate is the
    statistic on the full sample. ``statistic`` defaults to the mean (vectorised);
    pass e.g. ``lambda v: percentile(v, 95)`` for latency percentiles.
    Deterministic for a given ``seed``.
    """
    import numpy as np

    if len(values) == 0:
        raise ValueError("bootstrap_ci of empty sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_resamples, len(data)))
    if statistic is None:
        point = float(data.mean())
        stats = data[idx].mean(axis=1)
    else:
        point = float(statistic(data.tolist()))
        stats = np.array([statistic(row.tolist()) for row in data[idx]])
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return point, float(lower), float(upper)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) where a[i], b[i] are the same query.

    If the interval excludes 0 the difference is unlikely to be resampling noise.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n_resamples=n_resamples, confidence=confidence, seed=seed)
