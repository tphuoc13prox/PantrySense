from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.recipes.schemas import RecipeDetail, RecipeSearchRequest, RecipeSearchResponse
from app.backend.recipes.service import RecipeSearchService
from app.backend.retrieval.semantic import SemanticIndexNotFoundError

router = APIRouter(prefix="/api/recipes", tags=["recipes"])


@router.post("/search", response_model=RecipeSearchResponse)
def search_recipes(request: RecipeSearchRequest) -> RecipeSearchResponse:
    service = RecipeSearchService()
    try:
        results = service.search(request.ingredients)
        return RecipeSearchResponse(recipes=results)
    except SemanticIndexNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e


@router.get("/{recipe_id}", response_model=RecipeDetail)
def get_recipe(recipe_id: int) -> RecipeDetail:
    service = RecipeSearchService()
    recipe = service.get_by_id(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found.")
    return recipe
