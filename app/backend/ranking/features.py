from __future__ import annotations

from typing import Any

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


def extract_features(
    summary: Any,
    recipe_detail: dict[str, Any] | None = None,
) -> list[float]:
    """Extract a 9-dimensional numerical feature vector for a query-recipe candidate.

    Features:
        0: semantic_score (Dense cosine similarity)
        1: bm25_score (Sparse lexical score)
        2: rrf_score (Reciprocal rank fusion score)
        3: coverage (Matched / Required ingredient ratio)
        4: matched_count (Number of available ingredients)
        5: missing_count (Number of missing ingredients)
        6: cooking_time (Estimated cooking time in minutes)
        7: difficulty_num (1.0=Easy, 2.0=Medium, 3.0=Hard)
        8: total_ingredients (Total required ingredients count)
    """
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
