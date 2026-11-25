from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backend.config import (
    get_bm25_index_path,
    get_database_path,
    get_match_threshold,
    get_ranker_model_path,
    get_ranking_mode,
    get_retrieval_mode,
    get_vector_ids_path,
    get_vector_index_path,
)
from app.backend.database.connection import get_connection
from app.backend.database.recipe_metadata import (
    METADATA_SELECT_COLUMNS,
    MetadataIndex,
    RecipeMetadata,
    compute_recipe_metadata,
    metadata_from_row,
)
from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes import dietary, nutrition, substitution
from app.backend.recipes.instructions import clean_and_segment_instructions
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
from app.backend.retrieval.bm25_store import BM25Store, preferred_index_path
from app.backend.retrieval.candidates import CandidateMatch
from app.backend.retrieval.filtering import AllowedIds
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


# ---------------------------------------------------------------------------
# Shared heavy components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchComponents:
    """Index-backed objects that are expensive to load and safe to share."""

    matcher: IngredientMatcher
    ranker: RecipeRanker
    ml_ranker: MLRecipeRanker
    retriever: SemanticRetriever
    hybrid_retriever: HybridRetriever
    bm25_store: BM25Store
    metadata_index: MetadataIndex


_components_lock = threading.Lock()
_components: dict[tuple[str, ...], SearchComponents] = {}


def _components_key(database_path: Path | None) -> tuple[str, ...]:
    return (
        str(database_path or get_database_path()),
        str(get_vector_index_path()),
        str(get_vector_ids_path()),
        str(get_bm25_index_path()),
        str(get_ranker_model_path()),
    )


def build_search_components(database_path: Path | None = None) -> SearchComponents:
    """Create a fresh, unloaded set of components for the current configuration."""
    retriever = SemanticRetriever()
    bm25_store = BM25Store()
    return SearchComponents(
        matcher=IngredientMatcher(),
        ranker=RecipeRanker(),
        ml_ranker=MLRecipeRanker(),
        retriever=retriever,
        hybrid_retriever=HybridRetriever(semantic_retriever=retriever, bm25_store=bm25_store),
        bm25_store=bm25_store,
        metadata_index=MetadataIndex(database_path or get_database_path()),
    )


def get_search_components(database_path: Path | None = None) -> SearchComponents:
    """Components shared by every service using the same database and index paths.

    The key is taken from the configuration at call time, so changing the
    PANTRYSENSE_* paths (tests do this) yields a separate set instead of
    silently reusing indices loaded from the old paths.
    """
    key = _components_key(database_path)
    found = _components.get(key)
    if found is not None:
        return found
    with _components_lock:
        found = _components.get(key)
        if found is None:
            found = _components[key] = build_search_components(database_path)
        return found


def reset_search_components() -> None:
    """Drop cached components, e.g. after the setup pipeline rebuilt the indices."""
    with _components_lock:
        _components.clear()


def create_search_service(**kwargs: Any) -> "RecipeSearchService":
    """Build a service; components not given explicitly come from the shared cache."""
    return RecipeSearchService(**kwargs)


@dataclass(frozen=True)
class RecipeSearchService:
    database_path: Path | None = None
    match_threshold: float | None = None
    retrieval_mode: str | None = None
    ranking_mode: str | None = None
    matcher: IngredientMatcher | None = None
    ranker: RecipeRanker | None = None
    ml_ranker: MLRecipeRanker | None = None
    retriever: SemanticRetriever | None = None
    hybrid_retriever: HybridRetriever | None = None
    bm25_store: BM25Store | None = None
    metadata_index: MetadataIndex | None = None

    def __post_init__(self) -> None:
        names = ("matcher", "ranker", "ml_ranker", "retriever", "hybrid_retriever", "bm25_store", "metadata_index")
        if all(getattr(self, name) is not None for name in names):
            return
        shared = get_search_components(self.database_path)

        hybrid = self.hybrid_retriever
        if hybrid is None:
            if self.retriever is not None or self.bm25_store is not None:
                hybrid = HybridRetriever(
                    semantic_retriever=self.retriever or shared.retriever,
                    bm25_store=self.bm25_store or shared.bm25_store,
                )
            else:
                hybrid = shared.hybrid_retriever
        resolved = {
            "matcher": self.matcher or shared.matcher,
            "ranker": self.ranker or shared.ranker,
            "ml_ranker": self.ml_ranker or shared.ml_ranker,
            "hybrid_retriever": hybrid,
            "retriever": self.retriever or hybrid.semantic_retriever,
            "bm25_store": self.bm25_store if self.bm25_store is not None else hybrid.bm25_store,
            "metadata_index": self.metadata_index or shared.metadata_index,
        }
        for name, value in resolved.items():
            object.__setattr__(self, name, value)

    def allowed_ids(
        self,
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category: str | None = None,
    ) -> AllowedIds | None:
        """Recipe ids passing the filters, from stored metadata (None if unfiltered)."""
        return self.metadata_index.allowed(filters, max_cooking_time, category)

    def search(
        self,
        raw_ingredients: list[str],
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category: str | None = None,
        include_substitutions: bool = False,
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

        if mode not in ("hybrid", "semantic", "lexical"):
            return self._search_rule_based(
                raw_ingredients, threshold, filters, max_cooking_time, category, include_substitutions
            )

        allowed = self.allowed_ids(filters, max_cooking_time, category)
        if allowed is not None and allowed.is_empty:
            return []

        if mode == "hybrid":
            candidates = self.hybrid_retriever.retrieve(raw_ingredients, allowed=allowed)
        elif mode == "semantic":
            candidates = self.retriever.retrieve(raw_ingredients, allowed=allowed)
        else:
            if not self._ensure_lexical_loaded():
                raise SemanticIndexNotFoundError(
                    "BM25 index not found. Please run 'python scripts/build_vector_index.py'."
                )
            query_str = build_query_representation(raw_ingredients)
            candidates = [
                CandidateMatch(recipe_id=r_id, bm25_score=score, retrieval_rank=rank)
                for r_id, score, rank in self.bm25_store.search(query_str, allowed=allowed)
            ]
        return self._process_candidates(
            raw_ingredients,
            candidates,
            threshold,
            filters,
            max_cooking_time,
            category,
            include_substitutions=include_substitutions,
        )

    def _ensure_lexical_loaded(self) -> bool:
        if self.bm25_store is self.hybrid_retriever.bm25_store:
            return self.hybrid_retriever.ensure_bm25_loaded()
        if not self.bm25_store.is_empty:
            return True
        path = preferred_index_path(self.hybrid_retriever.bm25_path)
        return path.exists() and self.bm25_store.load(path)

    def _summary(
        self,
        row: Any,
        recipe_ingredients: list[str],
        meta: RecipeMetadata,
        raw_ingredients: list[str],
        include_substitutions: bool,
        cand: CandidateMatch | None = None,
    ) -> RecipeSummary:
        match = self.matcher.match(raw_ingredients, recipe_ingredients)
        sub_models: list[MissingIngredientSubstitution] = []
        if include_substitutions and match.missing_ingredients:
            sub_models = _build_substitution_models(
                substitution.suggest_recipe_substitutions(match.missing_ingredients, available_pantry=raw_ingredients)
            )
        return RecipeSummary(
            id=row["id"],
            title=row["title"],
            matched_ingredients=match.matched_ingredients,
            missing_ingredients=match.missing_ingredients,
            matched_count=match.matched_count,
            required_count=match.required_count,
            coverage=round(match.coverage, 4),
            cooking_time=row["cooking_time"],
            category=row["category"],
            dietary_tags=meta.dietary_tags,
            allergens=meta.allergens,
            # Search results carry per-serving and total values; the
            # per-ingredient breakdown is only built for the detail view.
            nutrition=_build_nutrition_model(meta.nutrition_dict(include_breakdown=False)),
            substitutions=sub_models,
            semantic_score=cand.semantic_score if cand else None,
            bm25_score=cand.bm25_score if cand else None,
            rrf_score=cand.rrf_score if cand else None,
            retrieval_rank=cand.retrieval_rank if cand else None,
        )

    def _process_candidates(
        self,
        raw_ingredients: list[str],
        candidates: list[CandidateMatch],
        threshold: float,
        filters: DietaryFilters | None = None,
        max_cooking_time: int | None = None,
        category_filter: str | None = None,
        include_substitutions: bool = False,
    ) -> list[RecipeSummary]:
        """Turn retrieved candidates into ranked summaries.

        Candidates normally arrive pre-filtered; the filter check here only
        guards against metadata that changed after the filter index loaded.
        """
        if not candidates:
            return []

        candidate_map = {c.recipe_id: c for c in candidates}
        candidate_ids = list(candidate_map.keys())

        placeholders = ",".join("?" for _ in candidate_ids)
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT r.id, r.title, r.ingredients, r.ingredient_details, r.cooking_time, r.difficulty,
                       r.servings, r.category, {METADATA_SELECT_COLUMNS}
                FROM recipes r
                LEFT JOIN recipe_metadata m ON m.recipe_id = r.id
                WHERE r.id IN ({placeholders})
                """,
                candidate_ids,
            ).fetchall()

        matches: list[RecipeSummary] = []
        details_map: dict[int, dict[str, Any]] = {}

        for row in rows:
            recipe_id = row["id"]
            recipe_ingredients = json.loads(row["ingredients"] or "[]")
            meta = metadata_from_row(row) or compute_recipe_metadata(
                recipe_ingredients, row["servings"], json.loads(row["ingredient_details"] or "[]")
            )

            if not _satisfies_filters(
                dietary_tags=meta.dietary_tags,
                allergens=meta.allergens,
                cooking_time=row["cooking_time"],
                category=row["category"],
                filters=filters,
                max_cooking_time=max_cooking_time,
                category_filter=category_filter,
            ):
                continue

            matches.append(
                self._summary(
                    row, recipe_ingredients, meta, raw_ingredients, include_substitutions, candidate_map.get(recipe_id)
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
        include_substitutions: bool = False,
    ) -> list[RecipeSummary]:
        matches: list[RecipeSummary] = []

        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT r.id, r.title, r.ingredients, r.ingredient_details, r.cooking_time, r.servings,
                       r.category, {METADATA_SELECT_COLUMNS}
                FROM recipes r
                LEFT JOIN recipe_metadata m ON m.recipe_id = r.id
                ORDER BY r.title
                """
            ).fetchall()

        for row in rows:
            recipe_ingredients = json.loads(row["ingredients"] or "[]")
            meta = metadata_from_row(row) or compute_recipe_metadata(
                recipe_ingredients, row["servings"], json.loads(row["ingredient_details"] or "[]")
            )

            if not _satisfies_filters(
                dietary_tags=meta.dietary_tags,
                allergens=meta.allergens,
                cooking_time=row["cooking_time"],
                category=row["category"],
                filters=filters,
                max_cooking_time=max_cooking_time,
                category_filter=category_filter,
            ):
                continue

            matches.append(self._summary(row, recipe_ingredients, meta, raw_ingredients, include_substitutions))

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

        stored_details = json.loads(row["ingredient_details"] or "[]")
        ingredient_details = stored_details
        raw_ingredients = json.loads(row["ingredients"] or "[]")
        if not ingredient_details:
            ingredient_details = [
                {"name": ingredient, "quantity": None, "unit": None}
                for ingredient in raw_ingredients
            ]

        # A single recipe: compute everything fresh, including the
        # per-ingredient nutrition breakdown and substitutions.
        dietary_tags = dietary.classify_recipe_dietary(raw_ingredients)
        allergens = dietary.detect_recipe_allergens(raw_ingredients)
        raw_nut = nutrition.calculate_recipe_nutrition(
            raw_ingredients, servings=row["servings"], ingredient_details=stored_details or None
        )
        nutrition_model = _build_nutrition_model(raw_nut)
        raw_subs = substitution.suggest_recipe_substitutions(raw_ingredients)
        sub_models = _build_substitution_models(raw_subs)
        clean_instructions = clean_and_segment_instructions(row["instructions"])

        return RecipeDetail(
            id=row["id"],
            title=row["title"],
            ingredients=[IngredientDetail(**ingredient) for ingredient in ingredient_details],
            instructions=clean_instructions,
            cooking_time=row["cooking_time"],
            difficulty=row["difficulty"],
            servings=row["servings"],
            category=row["category"],
            dietary_tags=dietary_tags,
            allergens=allergens,
            nutrition=nutrition_model,
            substitutions=sub_models,
        )
