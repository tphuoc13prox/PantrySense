from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backend.config import get_match_threshold, get_ranking_mode, get_retrieval_mode
from app.backend.database.connection import get_connection
from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes import dietary, nutrition, substitution
from app.backend.recipes.matching import IngredientMatcher
from app.backend.recipes.ranking import RecipeRanker
from app.backend.recipes.schemas import (
    DietaryFilters,
    IngredientDetail,
    MissingIngredientSubstitution,
    NutritionBreakdownItem,
    NutritionInfo,
    NutritionPerServing,
    RecipeDetail,
    RecipeSummary,
    SubstitutionItem,
)
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.fusion import CandidateMatch
from app.backend.retrieval.hybrid import HybridRetriever
from app.backend.retrieval.semantic import SemanticIndexNotFoundError, SemanticRetriever, build_query_representation


def _build_nutrition_model(raw_nut: dict[str, Any]) -> NutritionInfo:
    return NutritionInfo(
        servings=raw_nut.get("servings", 4),
        per_serving=NutritionPerServing(**raw_nut.get("per_serving", {})),
        total=NutritionPerServing(**raw_nut.get("total", {})),
        breakdown=[NutritionBreakdownItem(**b) for b in raw_nut.get("breakdown", [])],
    )


def _build_substitution_models(raw_subs: list[dict[str, Any]]) -> list[MissingIngredientSubstitution]:
    result = []
    for s in raw_subs:
        items = [
            SubstitutionItem(
                substitute=sub["substitute"],
                ratio=sub["ratio"],
                context=sub["context"],
                in_pantry=sub.get("in_pantry", False),
            )
            for sub in s.get("substitutes", [])
        ]
        result.append(
            MissingIngredientSubstitution(
                missing_ingredient=s["missing_ingredient"],
                substitutes=items,
                has_pantry_match=s.get("has_pantry_match", False),
            )
        )
    return result


def _satisfies_filters(
    dietary_tags: dict[str, bool],
    allergens: list[str],
    cooking_time: int | None,
    category: str | None,
    filters: DietaryFilters | None,
    max_cooking_time: int | None = None,
    category_filter: str | None = None,
) -> bool:
    eff_max_time = max_cooking_time
    eff_cat = category_filter

    if filters:
        if filters.vegetarian and not dietary_tags.get("vegetarian", False):
            return False
        if filters.vegan and not dietary_tags.get("vegan", False):
            return False
        if filters.gluten_free and not dietary_tags.get("gluten_free", False):
            return False
        if filters.dairy_free and not dietary_tags.get("dairy_free", False):
            return False
        if filters.nut_free and not dietary_tags.get("nut_free", False):
            return False
        if filters.keto_low_carb and not dietary_tags.get("keto_low_carb", False):
            return False

        if filters.excluded_allergens:
            allergen_set = set(allergens)
            for excl in filters.excluded_allergens:
                if excl.lower().strip() in allergen_set:
                    return False

        if filters.max_cooking_time is not None:
            eff_max_time = filters.max_cooking_time
        if filters.category is not None:
            eff_cat = filters.category

    if eff_max_time is not None and cooking_time is not None:
        if cooking_time > eff_max_time:
            return False

    if eff_cat and eff_cat.strip():
        req_cat = eff_cat.strip().lower()
        if not category or req_cat not in category.lower():
            return False

    return True


@dataclass(frozen=True)
class RecipeSearchService:
    database_path: Path | None = None
    match_threshold: float | None = None
    retrieval_mode: str | None = None
    ranking_mode: str | None = None
    matcher: IngredientMatcher = IngredientMatcher()
    ranker: RecipeRanker = RecipeRanker()
    ml_ranker: MLRecipeRanker = MLRecipeRanker()
    retriever: SemanticRetriever = SemanticRetriever()
    hybrid_retriever: HybridRetriever = HybridRetriever()
    bm25_store: BM25Store = BM25Store()

    def search(
        self,
        raw_ingredients: list[str],
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category: str | None = None,
    ) -> list[RecipeSummary]:
        if not raw_ingredients:
            return []

        mode = self.retrieval_mode if self.retrieval_mode is not None else get_retrieval_mode()
        if self.match_threshold is not None:
            threshold = self.match_threshold
        elif len(raw_ingredients) <= 2:
            threshold = min(get_match_threshold(), 0.10)
        else:
            threshold = get_match_threshold()

        if mode == "hybrid":
            candidates = self.hybrid_retriever.retrieve(raw_ingredients)
            return self._process_candidates(raw_ingredients, candidates, threshold, filters, max_cooking_time, category)
        elif mode == "semantic":
            candidates = self.retriever.retrieve(raw_ingredients)
            return self._process_candidates(raw_ingredients, candidates, threshold, filters, max_cooking_time, category)
        elif mode == "lexical":
            if not self.bm25_store.is_empty or self.hybrid_retriever.bm25_path.exists():
                if self.bm25_store.is_empty:
                    self.bm25_store.load(self.hybrid_retriever.bm25_path)
                query_str = build_query_representation(raw_ingredients)
                raw_sparse = self.bm25_store.search(query_str)
                candidates = [
                    CandidateMatch(recipe_id=r_id, bm25_score=score, retrieval_rank=rank)
                    for r_id, score, rank in raw_sparse
                ]
                return self._process_candidates(raw_ingredients, candidates, threshold, filters, max_cooking_time, category)
            else:
                raise SemanticIndexNotFoundError(
                    "BM25 index not found. Please run 'python scripts/build_vector_index.py'."
                )
        return self._search_rule_based(raw_ingredients, threshold, filters, max_cooking_time, category)

    def _process_candidates(
        self,
        raw_ingredients: list[str],
        candidates: list[CandidateMatch],
        threshold: float,
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category_filter: str | None = None,
    ) -> list[RecipeSummary]:
        if not candidates:
            return []

        candidate_map = {c.recipe_id: c for c in candidates}
        candidate_ids = list(candidate_map.keys())

        placeholders = ",".join("?" for _ in candidate_ids)
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT id, title, ingredients, cooking_time, difficulty, servings, category
                FROM recipes
                WHERE id IN ({placeholders})
                """,
                candidate_ids,
            ).fetchall()

        matches: list[RecipeSummary] = []
        details_map: dict[int, dict[str, Any]] = {}

        for row in rows:
            recipe_id = row["id"]
            cand = candidate_map.get(recipe_id)
            recipe_ingredients = json.loads(row["ingredients"] or "[]")
            match = self.matcher.match(raw_ingredients, recipe_ingredients)

            dietary_tags = dietary.classify_recipe_dietary(recipe_ingredients)
            allergens = dietary.detect_recipe_allergens(recipe_ingredients)

            # Filter check
            if not _satisfies_filters(
                dietary_tags=dietary_tags,
                allergens=allergens,
                cooking_time=row["cooking_time"],
                category=row["category"],
                filters=filters,
                max_cooking_time=max_cooking_time,
                category_filter=category_filter,
            ):
                continue

            raw_nut = nutrition.calculate_recipe_nutrition(recipe_ingredients, servings=row["servings"])
            nutrition_model = _build_nutrition_model(raw_nut)

            raw_subs = substitution.suggest_recipe_substitutions(match.missing_ingredients, available_pantry=raw_ingredients)
            sub_models = _build_substitution_models(raw_subs)

            matches.append(
                RecipeSummary(
                    id=recipe_id,
                    title=row["title"],
                    matched_ingredients=match.matched_ingredients,
                    missing_ingredients=match.missing_ingredients,
                    matched_count=match.matched_count,
                    required_count=match.required_count,
                    coverage=round(match.coverage, 4),
                    cooking_time=row["cooking_time"],
                    category=row["category"],
                    dietary_tags=dietary_tags,
                    allergens=allergens,
                    nutrition=nutrition_model,
                    substitutions=sub_models,
                    semantic_score=cand.semantic_score if cand else None,
                    bm25_score=cand.bm25_score if cand else None,
                    rrf_score=cand.rrf_score if cand else None,
                    retrieval_rank=cand.retrieval_rank if cand else None,
                )
            )

            details_map[recipe_id] = {
                "cooking_time": row["cooking_time"],
                "difficulty": row["difficulty"],
                "ingredients": recipe_ingredients,
            }

        # Apply ML Ranking (or fallback to heuristic)
        r_mode = self.ranking_mode if self.ranking_mode is not None else get_ranking_mode()
        if r_mode == "ml":
            return self.ml_ranker.rank(matches, details_map=details_map, minimum_coverage=threshold)
        return self.ranker.rank(matches, threshold)

    def _search_rule_based(
        self,
        raw_ingredients: list[str],
        threshold: float,
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category_filter: str | None = None,
    ) -> list[RecipeSummary]:
        matches: list[RecipeSummary] = []

        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                "SELECT id, title, ingredients, cooking_time, servings, category FROM recipes ORDER BY title"
            ).fetchall()

        for row in rows:
            recipe_ingredients = json.loads(row["ingredients"] or "[]")
            dietary_tags = dietary.classify_recipe_dietary(recipe_ingredients)
            allergens = dietary.detect_recipe_allergens(recipe_ingredients)

            if not _satisfies_filters(
                dietary_tags=dietary_tags,
                allergens=allergens,
                cooking_time=row["cooking_time"],
                category=row["category"],
                filters=filters,
                max_cooking_time=max_cooking_time,
                category_filter=category_filter,
            ):
                continue

            match = self.matcher.match(raw_ingredients, recipe_ingredients)
            raw_nut = nutrition.calculate_recipe_nutrition(recipe_ingredients, servings=row["servings"])
            nutrition_model = _build_nutrition_model(raw_nut)
            raw_subs = substitution.suggest_recipe_substitutions(match.missing_ingredients, available_pantry=raw_ingredients)
            sub_models = _build_substitution_models(raw_subs)

            matches.append(
                RecipeSummary(
                    id=row["id"],
                    title=row["title"],
                    matched_ingredients=match.matched_ingredients,
                    missing_ingredients=match.missing_ingredients,
                    matched_count=match.matched_count,
                    required_count=match.required_count,
                    coverage=round(match.coverage, 4),
                    cooking_time=row["cooking_time"],
                    category=row["category"],
                    dietary_tags=dietary_tags,
                    allergens=allergens,
                    nutrition=nutrition_model,
                    substitutions=sub_models,
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
        raw_ingredients = json.loads(row["ingredients"] or "[]")
        if not ingredient_details:
            ingredient_details = [
                {"name": ingredient, "quantity": None, "unit": None}
                for ingredient in raw_ingredients
            ]

        dietary_tags = dietary.classify_recipe_dietary(raw_ingredients)
        allergens = dietary.detect_recipe_allergens(raw_ingredients)
        raw_nut = nutrition.calculate_recipe_nutrition(raw_ingredients, servings=row["servings"])
        nutrition_model = _build_nutrition_model(raw_nut)
        raw_subs = substitution.suggest_recipe_substitutions(raw_ingredients)
        sub_models = _build_substitution_models(raw_subs)

        return RecipeDetail(
            id=row["id"],
            title=row["title"],
            ingredients=[IngredientDetail(**ingredient) for ingredient in ingredient_details],
            instructions=json.loads(row["instructions"] or "[]"),
            cooking_time=row["cooking_time"],
            difficulty=row["difficulty"],
            servings=row["servings"],
            category=row["category"],
            dietary_tags=dietary_tags,
            allergens=allergens,
            nutrition=nutrition_model,
            substitutions=sub_models,
        )
