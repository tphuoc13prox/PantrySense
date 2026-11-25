"""Parsing of raw dataset rows (Hugging Face food-recipes) into recipe dicts."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("pantrysense.setup")


def _parse_c_vector(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("c(") and val.endswith(")"):
            items = re.findall(r'"([^"]*)"', val) or re.findall(r"'([^']*)'", val)
            return [it.strip() for it in items if it.strip()]
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return [line.strip() for line in val.split("\n") if line.strip()]
    return []


def _parse_duration_minutes(val: Any) -> int:
    if not val or not isinstance(val, str):
        return 20
    hours = re.search(r"(\d+)H", val)
    mins = re.search(r"(\d+)M", val)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    return min(max(total, 5), 180) if total > 0 else 20


def _parse_quantity(val: Any) -> float:
    if not val:
        return 1.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if not val_str:
        return 1.0
    if "/" in val_str:
        parts = val_str.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                return num / den if den != 0 else 1.0
            except Exception:
                pass
    try:
        return float(val_str)
    except Exception:
        return 1.0


def _parse_hf_recipe_text(row_dict: dict[str, Any]) -> dict[str, Any] | None:
    try:
        title = (
            row_dict.get("Name")
            or row_dict.get("title")
            or row_dict.get("name")
            or row_dict.get("recipe_name")
        )
        if not title or not isinstance(title, str) or len(title.strip()) < 3:
            return None
        title = title.strip()

        # Parse ingredients from parts or lists
        raw_ingredients = (
            row_dict.get("RecipeIngredientParts")
            or row_dict.get("ingredients")
            or row_dict.get("NER")
            or row_dict.get("ingredient_list")
            or []
        )
        extracted_ingredients = _parse_c_vector(raw_ingredients)
        parsed_ingredients: list[str] = []
        ingredient_details: list[dict[str, Any]] = []

        for item in extracted_ingredients:
            clean_name = re.sub(r"[^\w\s-]", "", item.lower()).strip()
            if clean_name and len(clean_name) > 1:
                words = clean_name.split()
                short_name = " ".join(words[:3])
                if short_name not in parsed_ingredients:
                    parsed_ingredients.append(short_name)
                    ingredient_details.append({"name": short_name, "quantity": 1.0, "unit": "portion"})

        if not parsed_ingredients:
            return None

        # Parse instructions
        raw_instructions = (
            row_dict.get("RecipeInstructions")
            or row_dict.get("instructions")
            or row_dict.get("directions")
            or row_dict.get("steps")
            or []
        )
        parsed_instructions = _parse_c_vector(raw_instructions)
        if not parsed_instructions:
            parsed_instructions = [f"Prepare ingredients and cook {title} to desired taste."]

        cook_time_val = row_dict.get("CookTime") or row_dict.get("cooking_time") or row_dict.get("total_time")
        if isinstance(cook_time_val, str) and ("PT" in cook_time_val or "H" in cook_time_val or "M" in cook_time_val):
            cooking_time = _parse_duration_minutes(cook_time_val)
        else:
            try:
                cooking_time = int(cook_time_val or 20)
            except Exception:
                cooking_time = 20

        raw_servings = row_dict.get("RecipeServings") or row_dict.get("servings") or 2
        try:
            servings = int(raw_servings)
        except Exception:
            servings = 2

        raw_cat = str(row_dict.get("RecipeCategory") or row_dict.get("category") or "dinner").lower()
        if "breakfast" in raw_cat:
            category = "breakfast"
        elif "lunch" in raw_cat:
            category = "lunch"
        elif "soup" in raw_cat:
            category = "soup"
        elif "dessert" in raw_cat or "pie" in raw_cat or "cookie" in raw_cat:
            category = "dessert"
        elif "beverage" in raw_cat or "drink" in raw_cat:
            category = "breakfast"
        else:
            category = "dinner"

        return {
            "title": title,
            "ingredients": parsed_ingredients[:12],
            "ingredient_details": ingredient_details[:12],
            "instructions": parsed_instructions[:10],
            "cooking_time": min(max(cooking_time, 5), 180),
            "difficulty": "medium" if cooking_time > 25 else "easy",
            "servings": min(max(servings, 1), 12),
            "category": category,
        }
    except Exception as e:
        logger.debug("Skipping unparseable HF recipe row: %s", e)
        return None
