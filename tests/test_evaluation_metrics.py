from __future__ import annotations

import math

import pytest

from app.backend.evaluation.metrics import (
    bootstrap_ci,
    dcg_at_k,
    dedupe,
    first_relevant_rank,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


RANKED = [10, 20, 30, 40, 50]


def test_dedupe_keeps_first_occurrence() -> None:
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(RANKED, {30}) == 3
    assert first_relevant_rank(RANKED, {50, 20}) == 2
    assert first_relevant_rank(RANKED, {99}) is None
    assert first_relevant_rank([], {1}) is None
    # Duplicates do not consume ranks.
    assert first_relevant_rank([10, 10, 30], {30}) == 2


def test_hit_at_k() -> None:
    assert hit_at_k(RANKED, {30}, 3) == 1.0
    assert hit_at_k(RANKED, {30}, 2) == 0.0
    assert hit_at_k(RANKED, {10}, 1) == 1.0
    assert hit_at_k(RANKED, set(), 5) == 0.0


def test_recall_at_k_single_and_multi() -> None:
    assert recall_at_k(RANKED, {40}, 3) == 0.0
    assert recall_at_k(RANKED, {40}, 4) == 1.0
    assert recall_at_k(RANKED, {20, 40, 99}, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 5) == 0.0
    # k larger than the list is fine.
    assert recall_at_k(RANKED, {50}, 100) == 1.0


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, {10, 30}, 4) == pytest.approx(0.5)
    assert precision_at_k([10], {10}, 5) == pytest.approx(0.2)


def test_reciprocal_rank_and_mrr() -> None:
    assert reciprocal_rank(RANKED, {30}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {99}) == 0.0
    assert mrr_at_k(RANKED, {30}, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(RANKED, {30}, 2) == 0.0
    assert mrr_at_k(RANKED, {20, 30}, 10) == pytest.approx(0.5)


def test_ndcg_single_relevant() -> None:
    assert ndcg_at_k(RANKED, {10}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {20}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(RANKED, {50}, 4) == 0.0
    assert ndcg_at_k(RANKED, set(), 10) == 0.0


def test_ndcg_multi_and_graded() -> None:
    # Two relevant docs at ranks 1 and 3; ideal places them at 1 and 2.
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, {10, 30}, 5) == pytest.approx(expected)
    # Graded: perfect ordering by gain gives 1.0.
    assert ndcg_at_k([1, 2, 3], {1: 3, 2: 2, 3: 1}, 3) == pytest.approx(1.0)
    # Reversed ordering is strictly worse.
    assert ndcg_at_k([3, 2, 1], {1: 3, 2: 2, 3: 1}, 3) < 1.0
    assert dcg_at_k([1, 2], {1: 3, 2: 1}, 2) == pytest.approx(3 + 1 / math.log2(3))


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RANKED, {10}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, {10}, -1)


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile([5.0], 95) == 5.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    values = [0.0] * 50 + [1.0] * 50
    point, lo, hi = bootstrap_ci(values, n_resamples=500, seed=7)
    assert point == pytest.approx(0.5)
    assert lo < 0.5 < hi
    assert 0.3 < lo and hi < 0.7
    assert bootstrap_ci(values, n_resamples=500, seed=7) == (point, lo, hi)


def test_bootstrap_ci_constant_sample_has_zero_width() -> None:
    assert bootstrap_ci([0.25] * 20, n_resamples=200) == (0.25, 0.25, 0.25)


def test_bootstrap_ci_custom_statistic() -> None:
    values = [float(i) for i in range(1, 101)]
    point, lo, hi = bootstrap_ci(values, statistic=lambda v: percentile(v, 50), n_resamples=300)
    assert point == pytest.approx(50.5)
    assert lo <= point <= hi


def test_paired_bootstrap_diff() -> None:
    a = [1.0] * 30 + [0.0] * 10
    b = [0.0] * 30 + [0.0] * 10
    diff, lo, hi = paired_bootstrap_diff(a, b, n_resamples=500)
    assert diff == pytest.approx(0.75)
    assert lo > 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 2.0])
