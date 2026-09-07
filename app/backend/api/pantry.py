from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from app.backend.database import pantry_db
from app.backend.recipes.schemas import (
    PantryItemCreate,
    PantryItemResponse,
    PantryItemUpdate,
)

router = APIRouter(prefix="/api/pantry", tags=["pantry"])


@router.get("/items", response_model=list[PantryItemResponse])
def list_pantry_items() -> list[PantryItemResponse]:
    """Retrieves all items stored in the user's virtual pantry."""
    items = pantry_db.get_pantry_items()
    return [PantryItemResponse(**item) for item in items]


@router.post("/items", response_model=PantryItemResponse)
def add_pantry_item(payload: PantryItemCreate) -> PantryItemResponse:
    """Adds a new ingredient to the user's virtual pantry."""
    item = pantry_db.add_pantry_item(
        name=payload.name,
        quantity=payload.quantity,
        unit=payload.unit,
        category=payload.category,
        expiry_date=payload.expiry_date,
    )
    return PantryItemResponse(**item)


@router.put("/items/{item_id}", response_model=PantryItemResponse)
def update_pantry_item(item_id: int, payload: PantryItemUpdate) -> PantryItemResponse:
    """Updates an existing pantry item."""
    updated = pantry_db.update_pantry_item(
        item_id=item_id,
        name=payload.name,
        quantity=payload.quantity,
        unit=payload.unit,
        category=payload.category,
        expiry_date=payload.expiry_date,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Pantry item not found")
    return PantryItemResponse(**updated)


@router.delete("/items/{item_id}")
def delete_pantry_item(item_id: int) -> dict[str, bool]:
    """Deletes an item from the virtual pantry."""
    deleted = pantry_db.delete_pantry_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pantry item not found")
    return {"success": True}


@router.get("/expiring", response_model=list[PantryItemResponse])
def get_expiring_items(
    days: int = Query(default=3, ge=1, le=30)
) -> list[PantryItemResponse]:
    """Retrieves items that are expired or expiring within `days` threshold."""
    items = pantry_db.get_expiring_pantry_items(days_threshold=days)
    return [PantryItemResponse(**item) for item in items]
