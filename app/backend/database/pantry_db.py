from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from app.backend.database.connection import get_connection


def ensure_pantry_tables(database_path: Path | None = None) -> None:
    """Initializes tables for pantry items, favorites, and meal planning."""
    with get_connection(database_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pantry_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                quantity REAL NOT NULL DEFAULT 1.0,
                unit TEXT NOT NULL DEFAULT 'pcs',
                category TEXT NOT NULL DEFAULT 'General',
                expiry_date TEXT,
                added_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS favorite_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER NOT NULL,
                recipe_title TEXT NOT NULL,
                recipe_data TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(recipe_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meal_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT NOT NULL,
                meal_slot TEXT NOT NULL,
                recipe_id INTEGER NOT NULL,
                recipe_title TEXT NOT NULL,
                servings INTEGER NOT NULL DEFAULT 2,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(day_of_week, meal_slot)
            )
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Pantry Item Operations
# ---------------------------------------------------------------------------

def calculate_expiry_status(expiry_date_str: str | None) -> tuple[int | None, str]:
    """Calculates days until expiry and a categorical status (fresh, expiring_soon, expired)."""
    if not expiry_date_str:
        return None, "fresh"

    try:
        exp_date = datetime.date.fromisoformat(expiry_date_str.split("T")[0])
        today = datetime.date.today()
        days_left = (exp_date - today).days

        if days_left <= 0:
            return days_left, "expired"
        elif days_left <= 3:
            return days_left, "expiring_soon"
        else:
            return days_left, "fresh"
    except Exception:
        return None, "fresh"


def add_pantry_item(
    name: str,
    quantity: float = 1.0,
    unit: str = "pcs",
    category: str = "General",
    expiry_date: str | None = None,
    database_path: Path | None = None,
) -> dict[str, Any]:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO pantry_items (name, quantity, unit, category, expiry_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name.strip(), quantity, unit.strip(), category.strip(), expiry_date),
        )
        item_id = cursor.lastrowid
        conn.commit()

    days_left, status = calculate_expiry_status(expiry_date)
    return {
        "id": item_id,
        "name": name.strip(),
        "quantity": quantity,
        "unit": unit.strip(),
        "category": category.strip(),
        "expiry_date": expiry_date,
        "days_left": days_left,
        "status": status,
    }


def get_pantry_items(database_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        rows = conn.execute(
            "SELECT * FROM pantry_items ORDER BY expiry_date ASC, name ASC"
        ).fetchall()

    items = []
    for r in rows:
        days_left, status = calculate_expiry_status(r["expiry_date"])
        items.append({
            "id": r["id"],
            "name": r["name"],
            "quantity": r["quantity"],
            "unit": r["unit"],
            "category": r["category"],
            "expiry_date": r["expiry_date"],
            "added_date": r["added_date"],
            "days_left": days_left,
            "status": status,
        })
    return items


def update_pantry_item(
    item_id: int,
    name: str | None = None,
    quantity: float | None = None,
    unit: str | None = None,
    category: str | None = None,
    expiry_date: str | None = None,
    database_path: Path | None = None,
) -> dict[str, Any] | None:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        row = conn.execute("SELECT * FROM pantry_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None

        new_name = name.strip() if name is not None else row["name"]
        new_qty = quantity if quantity is not None else row["quantity"]
        new_unit = unit.strip() if unit is not None else row["unit"]
        new_cat = category.strip() if category is not None else row["category"]
        new_exp = expiry_date if expiry_date is not None else row["expiry_date"]

        conn.execute(
            """
            UPDATE pantry_items
            SET name = ?, quantity = ?, unit = ?, category = ?, expiry_date = ?
            WHERE id = ?
            """,
            (new_name, new_qty, new_unit, new_cat, new_exp, item_id),
        )
        conn.commit()

    days_left, status = calculate_expiry_status(new_exp)
    return {
        "id": item_id,
        "name": new_name,
        "quantity": new_qty,
        "unit": new_unit,
        "category": new_cat,
        "expiry_date": new_exp,
        "days_left": days_left,
        "status": status,
    }


def delete_pantry_item(item_id: int, database_path: Path | None = None) -> bool:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        cursor = conn.execute("DELETE FROM pantry_items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_expiring_pantry_items(
    days_threshold: int = 3,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Returns all items that are either already expired or expiring within threshold days."""
    all_items = get_pantry_items(database_path)
    return [
        item for item in all_items
        if item["status"] in ("expired", "expiring_soon")
    ]


# ---------------------------------------------------------------------------
# Favorites Operations
# ---------------------------------------------------------------------------

def add_favorite(
    recipe_id: int,
    recipe_title: str,
    recipe_data: dict[str, Any] | None = None,
    database_path: Path | None = None,
) -> dict[str, Any]:
    ensure_pantry_tables(database_path)
    data_str = json.dumps(recipe_data or {})
    with get_connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO favorite_recipes (recipe_id, recipe_title, recipe_data)
            VALUES (?, ?, ?)
            ON CONFLICT(recipe_id) DO UPDATE SET
                recipe_title = excluded.recipe_title,
                recipe_data = excluded.recipe_data
            """,
            (recipe_id, recipe_title, data_str),
        )
        conn.commit()
    return {"recipe_id": recipe_id, "recipe_title": recipe_title, "is_favorite": True}


def remove_favorite(recipe_id: int, database_path: Path | None = None) -> bool:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        cursor = conn.execute("DELETE FROM favorite_recipes WHERE recipe_id = ?", (recipe_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_favorites(database_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        rows = conn.execute(
            "SELECT * FROM favorite_recipes ORDER BY created_at DESC"
        ).fetchall()

    result = []
    for r in rows:
        try:
            parsed = json.loads(r["recipe_data"])
        except Exception:
            parsed = {}
        result.append({
            "id": r["id"],
            "recipe_id": r["recipe_id"],
            "recipe_title": r["recipe_title"],
            "recipe_data": parsed,
            "created_at": r["created_at"],
        })
    return result


def is_recipe_favorite(recipe_id: int, database_path: Path | None = None) -> bool:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        row = conn.execute("SELECT 1 FROM favorite_recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# Meal Planning Operations
# ---------------------------------------------------------------------------

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEAL_SLOTS = ["breakfast", "lunch", "dinner", "snack"]


def set_meal_plan_slot(
    day_of_week: str,
    meal_slot: str,
    recipe_id: int,
    recipe_title: str,
    servings: int = 2,
    database_path: Path | None = None,
) -> dict[str, Any]:
    ensure_pantry_tables(database_path)
    normalized_day = day_of_week.capitalize()
    normalized_slot = meal_slot.lower()

    with get_connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO meal_plans (day_of_week, meal_slot, recipe_id, recipe_title, servings)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(day_of_week, meal_slot) DO UPDATE SET
                recipe_id = excluded.recipe_id,
                recipe_title = excluded.recipe_title,
                servings = excluded.servings
            """,
            (normalized_day, normalized_slot, recipe_id, recipe_title, servings),
        )
        conn.commit()

    return {
        "day_of_week": normalized_day,
        "meal_slot": normalized_slot,
        "recipe_id": recipe_id,
        "recipe_title": recipe_title,
        "servings": servings,
    }


def get_weekly_meal_plan(database_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT mp.*, r.ingredients, r.instructions, r.cooking_time, r.category
            FROM meal_plans mp
            LEFT JOIN recipes r ON mp.recipe_id = r.id
            ORDER BY mp.id ASC
            """
        ).fetchall()

    plans = []
    for r in rows:
        ing_list = []
        if r["ingredients"]:
            try:
                ing_list = json.loads(r["ingredients"])
            except Exception:
                ing_list = []

        plans.append({
            "id": r["id"],
            "day_of_week": r["day_of_week"],
            "meal_slot": r["meal_slot"],
            "recipe_id": r["recipe_id"],
            "recipe_title": r["recipe_title"],
            "servings": r["servings"],
            "cooking_time": r["cooking_time"],
            "category": r["category"],
            "ingredients": ing_list,
        })
    return plans


def remove_meal_plan_slot(
    day_of_week: str,
    meal_slot: str,
    database_path: Path | None = None,
) -> bool:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM meal_plans WHERE day_of_week = ? AND meal_slot = ?",
            (day_of_week.capitalize(), meal_slot.lower()),
        )
        conn.commit()
        return cursor.rowcount > 0


def clear_all_meal_plans(database_path: Path | None = None) -> bool:
    ensure_pantry_tables(database_path)
    with get_connection(database_path) as conn:
        conn.execute("DELETE FROM meal_plans")
        conn.commit()
        return True


def generate_shopping_list(
    subtract_pantry: bool = True,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generates an aggregated shopping list from all planned meals for the week,

    optionally deducting ingredients already available in the virtual pantry.
    """
    plans = get_weekly_meal_plan(database_path)
    needed_ingredients: dict[str, int] = {}

    for plan in plans:
        for ing in plan["ingredients"]:
            norm = ing.strip().lower()
            if not norm:
                continue
            needed_ingredients[norm] = needed_ingredients.get(norm, 0) + 1

    pantry_items = get_pantry_items(database_path) if subtract_pantry else []
    pantry_names = {item["name"].strip().lower() for item in pantry_items}

    shopping_list = []
    for ing, recipe_count in sorted(needed_ingredients.items(), key=lambda x: x[0]):
        # Check if already in pantry
        in_stock = any(p in ing or ing in p for p in pantry_names)
        shopping_list.append({
            "ingredient": ing,
            "recipe_count": recipe_count,
            "in_stock": in_stock,
            "suggested_buy": not in_stock,
        })

    return shopping_list
