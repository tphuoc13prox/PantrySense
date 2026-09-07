from __future__ import annotations

import re
from typing import Any

# Baseline nutritional profiles per standard culinary portion (in grams/macros approx per ingredient entry)
# [calories, protein_g, carbs_g, fat_g, fiber_g]
NUTRITION_DATABASE: dict[str, dict[str, float]] = {
    # Proteins & Meats
    "chicken breast": {"calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0},
    "chicken": {"calories": 190, "protein": 27.0, "carbs": 0.0, "fat": 8.0, "fiber": 0.0},
    "beef": {"calories": 250, "protein": 26.0, "carbs": 0.0, "fat": 15.0, "fiber": 0.0},
    "ground beef": {"calories": 260, "protein": 25.0, "carbs": 0.0, "fat": 17.0, "fiber": 0.0},
    "pork": {"calories": 240, "protein": 27.0, "carbs": 0.0, "fat": 14.0, "fiber": 0.0},
    "bacon": {"calories": 140, "protein": 9.0, "carbs": 0.4, "fat": 11.0, "fiber": 0.0},
    "turkey": {"calories": 135, "protein": 30.0, "carbs": 0.0, "fat": 1.0, "fiber": 0.0},
    "salmon": {"calories": 208, "protein": 22.0, "carbs": 0.0, "fat": 13.0, "fiber": 0.0},
    "tuna": {"calories": 130, "protein": 28.0, "carbs": 0.0, "fat": 1.0, "fiber": 0.0},
    "shrimp": {"calories": 85, "protein": 18.0, "carbs": 0.2, "fat": 1.0, "fiber": 0.0},
    "fish": {"calories": 120, "protein": 20.0, "carbs": 0.0, "fat": 4.0, "fiber": 0.0},
    "egg": {"calories": 72, "protein": 6.3, "carbs": 0.4, "fat": 4.8, "fiber": 0.0},
    "tofu": {"calories": 80, "protein": 8.0, "carbs": 2.0, "fat": 4.5, "fiber": 1.0},
    "beans": {"calories": 120, "protein": 8.0, "carbs": 21.0, "fat": 0.5, "fiber": 7.0},
    "lentils": {"calories": 115, "protein": 9.0, "carbs": 20.0, "fat": 0.4, "fiber": 8.0},

    # Dairy & Eggs
    "milk": {"calories": 60, "protein": 3.4, "carbs": 5.0, "fat": 3.3, "fiber": 0.0},
    "butter": {"calories": 102, "protein": 0.1, "carbs": 0.0, "fat": 11.5, "fiber": 0.0},
    "cheese": {"calories": 110, "protein": 7.0, "carbs": 1.0, "fat": 9.0, "fiber": 0.0},
    "cheddar": {"calories": 115, "protein": 7.0, "carbs": 0.5, "fat": 9.5, "fiber": 0.0},
    "parmesan": {"calories": 110, "protein": 10.0, "carbs": 0.9, "fat": 7.3, "fiber": 0.0},
    "heavy cream": {"calories": 100, "protein": 0.8, "carbs": 0.8, "fat": 10.5, "fiber": 0.0},
    "sour cream": {"calories": 60, "protein": 0.7, "carbs": 1.4, "fat": 5.8, "fiber": 0.0},
    "yogurt": {"calories": 60, "protein": 5.0, "carbs": 7.0, "fat": 1.5, "fiber": 0.0},
    "greek yogurt": {"calories": 100, "protein": 10.0, "carbs": 4.0, "fat": 4.0, "fiber": 0.0},

    # Grains & Starches
    "flour": {"calories": 110, "protein": 3.0, "carbs": 23.0, "fat": 0.3, "fiber": 1.0},
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "fiber": 0.4},
    "pasta": {"calories": 160, "protein": 6.0, "carbs": 31.0, "fat": 0.9, "fiber": 1.8},
    "noodle": {"calories": 140, "protein": 5.0, "carbs": 26.0, "fat": 1.0, "fiber": 1.2},
    "bread": {"calories": 75, "protein": 2.5, "carbs": 14.0, "fat": 1.0, "fiber": 1.0},
    "potato": {"calories": 110, "protein": 2.5, "carbs": 26.0, "fat": 0.1, "fiber": 2.0},
    "potatoes": {"calories": 110, "protein": 2.5, "carbs": 26.0, "fat": 0.1, "fiber": 2.0},
    "oats": {"calories": 150, "protein": 5.0, "carbs": 27.0, "fat": 2.5, "fiber": 4.0},
    "corn": {"calories": 90, "protein": 3.0, "carbs": 19.0, "fat": 1.2, "fiber": 2.0},

    # Vegetables & Produce
    "onion": {"calories": 30, "protein": 0.8, "carbs": 7.0, "fat": 0.1, "fiber": 1.3},
    "garlic": {"calories": 10, "protein": 0.5, "carbs": 2.0, "fat": 0.0, "fiber": 0.1},
    "tomato": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fat": 0.2, "fiber": 1.5},
    "tomatoes": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fat": 0.2, "fiber": 1.5},
    "carrot": {"calories": 25, "protein": 0.6, "carbs": 6.0, "fat": 0.1, "fiber": 1.7},
    "carrots": {"calories": 25, "protein": 0.6, "carbs": 6.0, "fat": 0.1, "fiber": 1.7},
    "bell pepper": {"calories": 25, "protein": 1.0, "carbs": 6.0, "fat": 0.2, "fiber": 2.0},
    "pepper": {"calories": 20, "protein": 0.8, "carbs": 4.5, "fat": 0.2, "fiber": 1.5},
    "spinach": {"calories": 15, "protein": 1.9, "carbs": 2.4, "fat": 0.3, "fiber": 1.5},
    "broccoli": {"calories": 35, "protein": 2.8, "carbs": 7.0, "fat": 0.4, "fiber": 2.6},
    "mushroom": {"calories": 15, "protein": 2.2, "carbs": 2.3, "fat": 0.2, "fiber": 0.7},
    "mushrooms": {"calories": 15, "protein": 2.2, "carbs": 2.3, "fat": 0.2, "fiber": 0.7},
    "celery": {"calories": 10, "protein": 0.5, "carbs": 2.0, "fat": 0.1, "fiber": 1.0},
    "zucchini": {"calories": 20, "protein": 1.5, "carbs": 3.5, "fat": 0.4, "fiber": 1.2},
    "avocado": {"calories": 160, "protein": 2.0, "carbs": 8.5, "fat": 15.0, "fiber": 6.7},

    # Fats, Oils & Condiments
    "olive oil": {"calories": 119, "protein": 0.0, "carbs": 0.0, "fat": 13.5, "fiber": 0.0},
    "oil": {"calories": 120, "protein": 0.0, "carbs": 0.0, "fat": 14.0, "fiber": 0.0},
    "vegetable oil": {"calories": 120, "protein": 0.0, "carbs": 0.0, "fat": 14.0, "fiber": 0.0},
    "mayonnaise": {"calories": 90, "protein": 0.1, "carbs": 0.1, "fat": 10.0, "fiber": 0.0},
    "sugar": {"calories": 48, "protein": 0.0, "carbs": 12.5, "fat": 0.0, "fiber": 0.0},
    "honey": {"calories": 64, "protein": 0.1, "carbs": 17.3, "fat": 0.0, "fiber": 0.0},
    "soy sauce": {"calories": 10, "protein": 1.0, "carbs": 1.0, "fat": 0.0, "fiber": 0.0},
    "peanut butter": {"calories": 190, "protein": 8.0, "carbs": 7.0, "fat": 16.0, "fiber": 2.0},
    "nuts": {"calories": 160, "protein": 5.0, "carbs": 6.0, "fat": 14.0, "fiber": 2.0},
    "almonds": {"calories": 160, "protein": 6.0, "carbs": 6.0, "fat": 14.0, "fiber": 3.5},
    "walnuts": {"calories": 185, "protein": 4.3, "carbs": 3.9, "fat": 18.5, "fiber": 1.9},
}

# Generic fallback profile for unrecognized produce/seasonings
DEFAULT_NUTRITION_PROFILE = {"calories": 25.0, "protein": 0.8, "carbs": 3.5, "fat": 0.5, "fiber": 0.8}


def match_ingredient_nutrition(ingredient_raw: str) -> dict[str, float]:
    """Finds the best matching nutrient profile for an ingredient string."""
    clean = ingredient_raw.lower().strip()
    clean = re.sub(r"[^\w\s-]", "", clean)

    # 1. Exact match
    if clean in NUTRITION_DATABASE:
        return NUTRITION_DATABASE[clean]

    # 2. Key contained in ingredient string
    for key, data in NUTRITION_DATABASE.items():
        if key in clean:
            return data

    # 3. Token overlap match
    tokens = set(clean.split())
    for key, data in NUTRITION_DATABASE.items():
        if tokens & set(key.split()):
            return data

    return DEFAULT_NUTRITION_PROFILE


def calculate_recipe_nutrition(
    ingredients: list[str],
    servings: int | None = None,
) -> dict[str, Any]:
    """Estimates total and per-serving macronutrients for a list of recipe ingredients."""
    servings_count = servings if (servings and servings > 0) else 4  # Standard default

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    total_fiber = 0.0

    breakdown = []
    for ing in ingredients:
        if not ing or not ing.strip():
            continue
        data = match_ingredient_nutrition(ing)
        total_calories += data["calories"]
        total_protein += data["protein"]
        total_carbs += data["carbs"]
        total_fat += data["fat"]
        total_fiber += data["fiber"]
        breakdown.append({
            "ingredient": ing,
            "calories": round(data["calories"], 1),
            "protein": round(data["protein"], 1),
            "carbs": round(data["carbs"], 1),
            "fat": round(data["fat"], 1),
            "fiber": round(data["fiber"], 1),
        })

    # Nutrition per serving
    per_serving = {
        "calories": round(total_calories / servings_count, 1),
        "protein": round(total_protein / servings_count, 1),
        "carbs": round(total_carbs / servings_count, 1),
        "fat": round(total_fat / servings_count, 1),
        "fiber": round(total_fiber / servings_count, 1),
    }

    return {
        "servings": servings_count,
        "per_serving": per_serving,
        "total": {
            "calories": round(total_calories, 1),
            "protein": round(total_protein, 1),
            "carbs": round(total_carbs, 1),
            "fat": round(total_fat, 1),
            "fiber": round(total_fiber, 1),
        },
        "breakdown": breakdown,
    }
