from __future__ import annotations

from fastapi import APIRouter, Query

from app.backend.recipes.schemas import IngredientSuggestResponse, IngredientSuggestion
from app.backend.recipes.suggester import get_ingredient_suggester

router = APIRouter(prefix="/api/ingredients", tags=["ingredients"])


@router.get("/suggest", response_model=IngredientSuggestResponse)
def suggest_ingredients(
    q: str = Query(default="", description="Ingredient query prefix or typo string"),
    limit: int = Query(default=8, ge=1, le=25, description="Maximum suggestions to return"),
) -> IngredientSuggestResponse:
    suggester = get_ingredient_suggester()
    raw_suggestions = suggester.suggest(query=q, limit=limit)
    items: list[IngredientSuggestion] = []
    for item in raw_suggestions:
        if isinstance(item, IngredientSuggestion):
            items.append(item)
        else:
            items.append(IngredientSuggestion(**item))
    return IngredientSuggestResponse(
        query=q,
        suggestions=items,
    )
