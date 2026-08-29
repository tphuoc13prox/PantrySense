from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")

INGREDIENT_SYNONYMS = {
    "bell pepper": "pepper",
    "capsicum": "pepper",
    "tomatoes": "tomato",
    "eggs": "egg",
}


def normalize_ingredient(value: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", value.strip().lower())
    if not cleaned:
        return ""

    if cleaned in INGREDIENT_SYNONYMS:
        return INGREDIENT_SYNONYMS[cleaned]

    if cleaned.endswith("ies") and len(cleaned) > 3:
        return f"{cleaned[:-3]}y"

    if cleaned.endswith("oes") and len(cleaned) > 3:
        return cleaned[:-2]

    if cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
        return cleaned[:-1]

    return cleaned


def normalize_ingredients(values: list[str]) -> list[str]:
    normalized = {
        ingredient
        for value in values
        if (ingredient := normalize_ingredient(value))
    }
    return sorted(normalized)


@dataclass(frozen=True)
class IngredientMatch:
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    matched_count: int
    required_count: int
    coverage: float


class IngredientMatcher:
    def match(self, user_ingredients: list[str], recipe_ingredients: list[str]) -> IngredientMatch:
        normalized_user = set(normalize_ingredients(user_ingredients))
        normalized_recipe = normalize_ingredients(recipe_ingredients)
        recipe_set = set(normalized_recipe)

        matched = sorted(normalized_user.intersection(recipe_set))
        missing = [ingredient for ingredient in normalized_recipe if ingredient not in normalized_user]
        required_count = len(normalized_recipe)
        coverage = len(matched) / required_count if required_count else 0.0

        return IngredientMatch(
            matched_ingredients=matched,
            missing_ingredients=missing,
            matched_count=len(matched),
            required_count=required_count,
            coverage=coverage,
        )
