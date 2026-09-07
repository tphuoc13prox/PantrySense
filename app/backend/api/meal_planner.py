from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query
from app.backend.database import pantry_db
from app.backend.recipes.schemas import (
    FavoriteCreate,
    FavoriteResponse,
    MealPlanItemResponse,
    MealPlanSlotCreate,
    ShoppingListItemResponse,
)

router = APIRouter(prefix="/api", tags=["meal_planner"])


# ---------------------------------------------------------------------------
# Favorites Endpoints
# ---------------------------------------------------------------------------

@router.get("/favorites", response_model=list[FavoriteResponse])
def get_favorites() -> list[FavoriteResponse]:
    """Retrieves all bookmarked recipes."""
    favorites = pantry_db.get_favorites()
    return [
        FavoriteResponse(
            id=f["id"],
            recipe_id=f["recipe_id"],
            recipe_title=f["recipe_title"],
            recipe_data=f["recipe_data"],
            created_at=f["created_at"],
        )
        for f in favorites
    ]


@router.post("/favorites", response_model=dict[str, Any])
def add_favorite(payload: FavoriteCreate) -> dict[str, Any]:
    """Adds or updates a favorite recipe bookmark."""
    return pantry_db.add_favorite(
        recipe_id=payload.recipe_id,
        recipe_title=payload.recipe_title,
        recipe_data=payload.recipe_data,
    )


@router.delete("/favorites/{recipe_id}")
def remove_favorite(recipe_id: int) -> dict[str, bool]:
    """Removes a recipe from favorites."""
    deleted = pantry_db.remove_favorite(recipe_id)
    return {"success": deleted}


@router.get("/favorites/{recipe_id}/check")
def check_favorite(recipe_id: int) -> dict[str, bool]:
    """Checks if a recipe is marked as favorite."""
    is_fav = pantry_db.is_recipe_favorite(recipe_id)
    return {"is_favorite": is_fav}


# ---------------------------------------------------------------------------
# Meal Plan Endpoints
# ---------------------------------------------------------------------------

@router.get("/meal-plan", response_model=list[MealPlanItemResponse])
def get_weekly_meal_plan() -> list[MealPlanItemResponse]:
    """Retrieves the full weekly 7-day meal plan."""
    plans = pantry_db.get_weekly_meal_plan()
    return [MealPlanItemResponse(**p) for p in plans]


@router.post("/meal-plan", response_model=dict[str, Any])
def set_meal_plan_slot(payload: MealPlanSlotCreate) -> dict[str, Any]:
    """Assigns a recipe to a specific day of the week and meal slot."""
    return pantry_db.set_meal_plan_slot(
        day_of_week=payload.day_of_week,
        meal_slot=payload.meal_slot,
        recipe_id=payload.recipe_id,
        recipe_title=payload.recipe_title,
        servings=payload.servings,
    )


@router.delete("/meal-plan/{day_of_week}/{meal_slot}")
def remove_meal_plan_slot(day_of_week: str, meal_slot: str) -> dict[str, bool]:
    """Clears a specific slot in the weekly meal plan."""
    removed = pantry_db.remove_meal_plan_slot(day_of_week, meal_slot)
    return {"success": removed}


@router.delete("/meal-plan/clear")
def clear_all_meal_plans() -> dict[str, bool]:
    """Clears all scheduled meals from the weekly planner."""
    pantry_db.clear_all_meal_plans()
    return {"success": True}


@router.get("/meal-plan/shopping-list", response_model=list[ShoppingListItemResponse])
def get_shopping_list(
    subtract_pantry: bool = Query(default=True)
) -> list[ShoppingListItemResponse]:
    """Generates an aggregated shopping list from scheduled meals,

    optionally deducting ingredients present in the virtual pantry.
    """
    items = pantry_db.generate_shopping_list(subtract_pantry=subtract_pantry)
    return [ShoppingListItemResponse(**item) for item in items]
