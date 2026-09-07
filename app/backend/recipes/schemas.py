from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DietaryFilters(BaseModel):
    vegetarian: bool = False
    vegan: bool = False
    gluten_free: bool = False
    dairy_free: bool = False
    nut_free: bool = False
    keto_low_carb: bool = False
    excluded_allergens: list[str] = Field(default_factory=list)
    max_cooking_time: int | None = None
    category: str | None = None


class RecipeSearchRequest(BaseModel):
    ingredients: list[str] = Field(default_factory=list)
    filters: DietaryFilters | None = None
    max_cooking_time: int | None = None
    category: str | None = None


class NutritionPerServing(BaseModel):
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0


class NutritionBreakdownItem(BaseModel):
    ingredient: str
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0


class NutritionInfo(BaseModel):
    servings: int = 4
    per_serving: NutritionPerServing
    total: NutritionPerServing
    breakdown: list[NutritionBreakdownItem] = Field(default_factory=list)


class SubstitutionItem(BaseModel):
    substitute: str
    ratio: str
    context: str
    in_pantry: bool = False


class MissingIngredientSubstitution(BaseModel):
    missing_ingredient: str
    substitutes: list[SubstitutionItem] = Field(default_factory=list)
    has_pantry_match: bool = False


class RecipeSummary(BaseModel):
    id: int
    title: str
    matched_ingredients: list[str] = Field(default_factory=list)
    missing_ingredients: list[str] = Field(default_factory=list)
    matched_count: int = 0
    required_count: int = 0
    coverage: float = 0.0
    cooking_time: int | None = None
    category: str | None = None
    dietary_tags: dict[str, bool] = Field(default_factory=dict)
    allergens: list[str] = Field(default_factory=list)
    nutrition: NutritionInfo | None = None
    substitutions: list[MissingIngredientSubstitution] = Field(default_factory=list)
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
    dietary_tags: dict[str, bool] = Field(default_factory=dict)
    allergens: list[str] = Field(default_factory=list)
    nutrition: NutritionInfo | None = None
    substitutions: list[MissingIngredientSubstitution] = Field(default_factory=list)


class IngredientSuggestion(BaseModel):
    name: str
    frequency: int = 0
    match_type: str = "prefix"
    is_correction: bool = False
    confidence: float = 1.0


class IngredientSuggestResponse(BaseModel):
    query: str
    suggestions: list[IngredientSuggestion]


# ---------------------------------------------------------------------------
# Pantry Schemas
# ---------------------------------------------------------------------------

class PantryItemCreate(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "pcs"
    category: str = "General"
    expiry_date: str | None = None


class PantryItemUpdate(BaseModel):
    name: str | None = None
    quantity: float | None = None
    unit: str | None = None
    category: str | None = None
    expiry_date: str | None = None


class PantryItemResponse(BaseModel):
    id: int
    name: str
    quantity: float
    unit: str
    category: str
    expiry_date: str | None = None
    added_date: str | None = None
    days_left: int | None = None
    status: str = "fresh"


# ---------------------------------------------------------------------------
# Favorites & Meal Plan Schemas
# ---------------------------------------------------------------------------

class FavoriteCreate(BaseModel):
    recipe_id: int
    recipe_title: str
    recipe_data: dict[str, Any] = Field(default_factory=dict)


class FavoriteResponse(BaseModel):
    id: int
    recipe_id: int
    recipe_title: str
    recipe_data: dict[str, Any]
    created_at: str


class MealPlanSlotCreate(BaseModel):
    day_of_week: str
    meal_slot: str
    recipe_id: int
    recipe_title: str
    servings: int = 2


class MealPlanItemResponse(BaseModel):
    id: int
    day_of_week: str
    meal_slot: str
    recipe_id: int
    recipe_title: str
    servings: int = 2
    cooking_time: int | None = None
    category: str | None = None
    ingredients: list[str] = Field(default_factory=list)


class ShoppingListItemResponse(BaseModel):
    ingredient: str
    recipe_count: int
    in_stock: bool
    suggested_buy: bool


# ---------------------------------------------------------------------------
# Cooking Assistant Schemas
# ---------------------------------------------------------------------------

class ExtractedTimer(BaseModel):
    step_number: int
    minutes: int
    description: str


class CookingStepInfo(BaseModel):
    step_number: int
    instruction: str
    timer_minutes: int | None = None
    timer_description: str | None = None
