from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# Legacy 9-dim per-candidate vector. Kept so that models trained before the
# query-relative feature set (bare estimators in ranker_model.joblib) still load.
FEATURE_NAMES = [
    "semantic_score",
    "bm25_score",
    "rrf_score",
    "coverage",
    "matched_count",
    "missing_count",
    "cooking_time",
    "difficulty_num",
    "total_ingredients",
]

DIFFICULTY_MAP = {
    "easy": 1.0,
    "medium": 2.0,
    "hard": 3.0,
}

# Corpus median of recipes.cooking_time (315k recipes). Used when it is missing.
DEFAULT_COOKING_TIME = 20.0

# Normalised ingredients present in >= 9% of recipes in the corpus. They say
# little about which recipe a pantry list points to, so they are discounted in
# the "missing" count. Same threshold the synthetic query generator uses.
STAPLE_INGREDIENTS = frozenset(
    {
        "salt",
        "butter",
        "egg",
        "onion",
        "sugar",
        "water",
        "garlic clove",
        "olive oil",
        "milk",
        "flour",
        "pepper",
    }
)

# Rank given to a candidate that is absent from the BM25 or dense list.
ABSENT_RANK = 1000.0

LTR_FEATURE_NAMES = [
    # raw retrieval scores
    "semantic_score",
    "bm25_score",
    "rrf_score",
    "has_dense",
    "has_bm25",
    # ranks / scores relative to the other candidates of the same query
    "dense_rank",
    "bm25_rank",
    "fused_rank",
    "bm25_rel",
    "dense_gap",
    "rrf_rel",
    # ingredient match
    "coverage",
    "matched_count",
    "missing_count",
    "required_count",
    "missing_nonstaple_count",
    # query-level
    "query_size",
    "query_coverage",
    # title
    "title_overlap",
    "title_overlap_count",
    # recipe metadata
    "cooking_time",
    "cooking_time_missing",
    "difficulty_num",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "retrieval_scores": ["semantic_score", "bm25_score", "rrf_score", "has_dense", "has_bm25"],
    "list_relative": ["dense_rank", "bm25_rank", "fused_rank", "bm25_rel", "dense_gap", "rrf_rel"],
    "ingredient_match": ["coverage", "matched_count", "missing_count", "required_count", "missing_nonstaple_count"],
    "query_level": ["query_size", "query_coverage"],
    "title": ["title_overlap", "title_overlap_count"],
    "metadata": ["cooking_time", "cooking_time_missing", "difficulty_num"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def extract_features(
    summary: Any,
    recipe_detail: dict[str, Any] | None = None,
) -> list[float]:
    """Legacy 9-dim feature vector for one candidate (see FEATURE_NAMES)."""
    sem_score = float(getattr(summary, "semantic_score", 0.0) or 0.0)
    bm25_score = float(getattr(summary, "bm25_score", 0.0) or 0.0)
    rrf_score = float(getattr(summary, "rrf_score", 0.0) or 0.0)
    coverage = float(getattr(summary, "coverage", 0.0) or 0.0)
    matched_count = float(getattr(summary, "matched_count", 0) or 0)
    missing_ingredients = getattr(summary, "missing_ingredients", []) or []
    missing_count = float(len(missing_ingredients))

    cooking_time = 25.0
    difficulty_num = 2.0
    total_ingredients = float(getattr(summary, "required_count", matched_count + missing_count) or (matched_count + missing_count))

    if recipe_detail:
        if recipe_detail.get("cooking_time") is not None:
            cooking_time = float(recipe_detail["cooking_time"])
        diff_str = str(recipe_detail.get("difficulty") or "medium").strip().lower()
        difficulty_num = DIFFICULTY_MAP.get(diff_str, 2.0)
        if recipe_detail.get("ingredients"):
            total_ingredients = float(len(recipe_detail["ingredients"]))

    return [
        sem_score,
        bm25_score,
        rrf_score,
        coverage,
        matched_count,
        missing_count,
        cooking_time,
        difficulty_num,
        total_ingredients,
    ]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _dense_ranks(values: list[float | None]) -> list[float]:
    """1-based rank by descending value; None gets ABSENT_RANK."""
    present = sorted((i for i, v in enumerate(values) if v is not None), key=lambda i: -float(values[i]))
    ranks = [ABSENT_RANK] * len(values)
    for rank, idx in enumerate(present, start=1):
        ranks[idx] = float(rank)
    return ranks


def _cooking_time(summary: Any, detail: dict[str, Any] | None) -> tuple[float, float]:
    raw = detail.get("cooking_time") if detail else None
    if raw is None:
        raw = getattr(summary, "cooking_time", None)
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    if value is None or value <= 0:
        return DEFAULT_COOKING_TIME, 1.0
    return value, 0.0


def build_feature_matrix(
    summaries: Sequence[Any],
    details_map: dict[int, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Feature matrix (n_candidates x len(LTR_FEATURE_NAMES)) for one query.

    Features that compare a candidate with the rest of the list (ranks, ratios
    to the list max) are computed over exactly the candidates passed in, which
    at serving time are the ones that survived the coverage threshold.

    The ranker does not receive the raw query, so the query is approximated by
    the union of matched ingredients across candidates. Query ingredients that
    no candidate contains are invisible here; that is the same at train and
    serve time because both go through this function.
    """
    n = len(summaries)
    if n == 0:
        return np.zeros((0, len(LTR_FEATURE_NAMES)), dtype=np.float32)
    details_map = details_map or {}

    sem = [getattr(s, "semantic_score", None) for s in summaries]
    bm25 = [getattr(s, "bm25_score", None) for s in summaries]
    rrf = [float(getattr(s, "rrf_score", None) or 0.0) for s in summaries]

    dense_rank = _dense_ranks(sem)
    bm25_rank = _dense_ranks(bm25)
    max_bm25 = max((float(v) for v in bm25 if v is not None), default=0.0)
    max_sem = max((float(v) for v in sem if v is not None), default=0.0)
    max_rrf = max(rrf, default=0.0)

    query_ings: set[str] = set()
    for s in summaries:
        query_ings.update(getattr(s, "matched_ingredients", None) or [])
    query_size = float(len(query_ings))
    query_tokens: set[str] = set()
    for ing in query_ings:
        query_tokens |= _tokens(ing)

    rows = np.zeros((n, len(LTR_FEATURE_NAMES)), dtype=np.float32)
    for i, s in enumerate(summaries):
        detail = details_map.get(getattr(s, "id", None))
        matched = float(getattr(s, "matched_count", 0) or 0)
        missing = list(getattr(s, "missing_ingredients", None) or [])
        required = float(getattr(s, "required_count", 0) or 0) or matched + len(missing)
        missing_nonstaple = float(sum(1 for m in missing if m not in STAPLE_INGREDIENTS))

        title_tokens = _tokens(str(getattr(s, "title", "") or ""))
        overlap = len(title_tokens & query_tokens)
        title_overlap = overlap / len(title_tokens) if title_tokens else 0.0

        cooking_time, cooking_missing = _cooking_time(s, detail)
        diff_str = str((detail or {}).get("difficulty") or "medium").strip().lower()

        fused = getattr(s, "retrieval_rank", None)
        rows[i] = [
            float(sem[i] or 0.0),
            float(bm25[i] or 0.0),
            rrf[i],
            0.0 if sem[i] is None else 1.0,
            0.0 if bm25[i] is None else 1.0,
            dense_rank[i],
            bm25_rank[i],
            float(fused) if fused is not None else ABSENT_RANK,
            float(bm25[i]) / max_bm25 if bm25[i] is not None and max_bm25 > 0 else 0.0,
            max_sem - float(sem[i]) if sem[i] is not None else 1.0,
            rrf[i] / max_rrf if max_rrf > 0 else 0.0,
            float(getattr(s, "coverage", 0.0) or 0.0),
            matched,
            float(len(missing)),
            required,
            missing_nonstaple,
            query_size,
            matched / query_size if query_size > 0 else 0.0,
            title_overlap,
            float(overlap),
            cooking_time,
            cooking_missing,
            DIFFICULTY_MAP.get(diff_str, 2.0),
        ]
    return rows
