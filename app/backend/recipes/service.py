from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backend.config import get_match_threshold, get_ranking_mode, get_retrieval_mode
from app.backend.database.connection import get_connection
from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes.matching import IngredientMatcher
from app.backend.recipes.ranking import RecipeRanker
from app.backend.recipes.schemas import IngredientDetail, RecipeDetail, RecipeSummary
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.fusion import CandidateMatch
from app.backend.retrieval.hybrid import HybridRetriever
from app.backend.retrieval.semantic import SemanticIndexNotFoundError, SemanticRetriever, build_query_representation


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

    def search(self, raw_ingredients: list[str]) -> list[RecipeSummary]:
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
            return self._process_candidates(raw_ingredients, candidates, threshold)
        elif mode == "semantic":
            candidates = self.retriever.retrieve(raw_ingredients)
            return self._process_candidates(raw_ingredients, candidates, threshold)
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
                return self._process_candidates(raw_ingredients, candidates, threshold)
            else:
                raise SemanticIndexNotFoundError(
                    "BM25 index not found. Please run 'python scripts/build_vector_index.py'."
                )
        return self._search_rule_based(raw_ingredients, threshold)

    def _process_candidates(
        self,
        raw_ingredients: list[str],
        candidates: list[CandidateMatch],
        threshold: float,
    ) -> list[RecipeSummary]:
        if not candidates:
            return []

        candidate_map = {c.recipe_id: c for c in candidates}
        candidate_ids = list(candidate_map.keys())

        placeholders = ",".join("?" for _ in candidate_ids)
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT id, title, ingredients, cooking_time, difficulty
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
