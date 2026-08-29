from __future__ import annotations

from app.backend.recipes.schemas import RecipeSummary


class RecipeRanker:
    def rank(self, recipes: list[RecipeSummary], minimum_coverage: float) -> list[RecipeSummary]:
        filtered = [
            recipe for recipe in recipes
            if recipe.required_count > 0 and recipe.coverage >= minimum_coverage
        ]
        return sorted(
            filtered,
            key=lambda recipe: (
                -recipe.coverage,
                -recipe.matched_count,
                len(recipe.missing_ingredients),
                recipe.title,
            ),
        )
