from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.backend.config import get_match_threshold, get_retrieval_mode
from app.backend.database.connection import get_connection
from app.backend.recipes.matching import IngredientMatcher
from app.backend.recipes.ranking import RecipeRanker
from app.backend.recipes.schemas import IngredientDetail, RecipeDetail, RecipeSummary
from app.backend.retrieval.semantic import SemanticIndexNotFoundError, SemanticRetriever


@dataclass(frozen=True)
class RecipeSearchService:
    database_path: Path | None = None
    match_threshold: float | None = None
    retrieval_mode: str | None = None
    matcher: IngredientMatcher = IngredientMatcher()
    ranker: RecipeRanker = RecipeRanker()
    retriever: SemanticRetriever = SemanticRetriever()

    def search(self, raw_ingredients: list[str]) -> list[RecipeSummary]:
        if not raw_ingredients:
            return []

        mode = self.retrieval_mode if self.retrieval_mode is not None else get_retrieval_mode()
        threshold = self.match_threshold if self.match_threshold is not None else get_match_threshold()

        if mode == "semantic":
            return self._search_semantic(raw_ingredients, threshold)
        return self._search_rule_based(raw_ingredients, threshold)

    def _search_rule_based(self, raw_ingredients: list[str], threshold: float) -> list[RecipeSummary]:
        matches: list[RecipeSummary] = []

        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                "SELECT id, title, ingredients FROM recipes ORDER BY title"
            ).fetchall()

        for row in rows:
            recipe_ingredients = json.loads(row["ingredients"])
            match = self.matcher.match(raw_ingredients, recipe_ingredients)
            matches.append(
                RecipeSummary(
                    id=row["id"],
                    title=row["title"],
                    matched_ingredients=match.matched_ingredients,
                    missing_ingredients=match.missing_ingredients,
                    matched_count=match.matched_count,
                    required_count=match.required_count,
                    coverage=round(match.coverage, 4),
                )
            )

        return self.ranker.rank(matches, threshold)

    def _search_semantic(self, raw_ingredients: list[str], threshold: float) -> list[RecipeSummary]:
        # Retrieve Top-K semantic candidates
        candidates = self.retriever.retrieve(raw_ingredients)
        if not candidates:
            return []

        candidate_map = {c.recipe_id: c for c in candidates}
        candidate_ids = list(candidate_map.keys())

        placeholders = ",".join("?" for _ in candidate_ids)
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                f"SELECT id, title, ingredients FROM recipes WHERE id IN ({placeholders})",
                candidate_ids,
            ).fetchall()

        matches: list[RecipeSummary] = []
        for row in rows:
            recipe_id = row["id"]
            cand = candidate_map.get(recipe_id)
            recipe_ingredients = json.loads(row["ingredients"])
            match = self.matcher.match(raw_ingredients, recipe_ingredients)
            matches.append(
                RecipeSummary(
                    id=recipe_id,
                    title=row["title"],
                    matched_ingredients=match.matched_ingredients,
                    missing_ingredients=match.missing_ingredients,
                    matched_count=match.matched_count,
                    required_count=match.required_count,
                    coverage=round(match.coverage, 4),
                    semantic_score=cand.semantic_score if cand else None,
                    retrieval_rank=cand.retrieval_rank if cand else None,
                )
            )

        return self.ranker.rank(matches, threshold)

    def get_by_id(self, recipe_id: int) -> RecipeDetail | None:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    title,
                    ingredients,
                    ingredient_details,
                    instructions,
                    cooking_time,
                    difficulty,
                    servings,
                    category
                FROM recipes
                WHERE id = ?
                """,
                (recipe_id,),
            ).fetchone()

        if row is None:
            return None

        ingredient_details = json.loads(row["ingredient_details"] or "[]")
        if not ingredient_details:
            ingredient_details = [
                {"name": ingredient, "quantity": None, "unit": None}
                for ingredient in json.loads(row["ingredients"] or "[]")
            ]

        return RecipeDetail(
            id=row["id"],
            title=row["title"],
            ingredients=[IngredientDetail(**ingredient) for ingredient in ingredient_details],
            instructions=json.loads(row["instructions"] or "[]"),
            cooking_time=row["cooking_time"],
            difficulty=row["difficulty"],
            servings=row["servings"],
            category=row["category"],
        )
