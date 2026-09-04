from __future__ import annotations

from pydantic import BaseModel, Field


class RecipeSearchRequest(BaseModel):
    ingredients: list[str] = Field(default_factory=list)


class RecipeSummary(BaseModel):
    id: int
    title: str
    matched_ingredients: list[str] = Field(default_factory=list)
    missing_ingredients: list[str] = Field(default_factory=list)
    matched_count: int = 0
    required_count: int = 0
    coverage: float = 0.0
    semantic_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    ml_score: float | None = None
    retrieval_rank: int | None = None




class RecipeSearchResponse(BaseModel):
    recipes: list[RecipeSummary]


class IngredientDetail(BaseModel):
    name: str
    quantity: float | int | None = None
    unit: str | None = None


class RecipeDetail(BaseModel):
    id: int
    title: str
    ingredients: list[IngredientDetail]
    instructions: list[str]
    cooking_time: int | None = None
    difficulty: str | None = None
    servings: int | None = None
    category: str | None = None
