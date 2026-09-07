from __future__ import annotations

import re
from typing import Any

# Knowledge base mapping common ingredients to practical culinary substitutes
SUBSTITUTION_KNOWLEDGE_BASE: dict[str, list[dict[str, str]]] = {
    "butter": [
        {"substitute": "olive oil", "ratio": "3/4 cup oil for 1 cup butter", "context": "Cooking & baking (liquids/sauté)"},
        {"substitute": "coconut oil", "ratio": "1:1", "context": "Baking & high-heat cooking"},
        {"substitute": "applesauce", "ratio": "1:1", "context": "Low-fat baking (cakes/muffins)"},
        {"substitute": "margarine", "ratio": "1:1", "context": "General cooking and baking"},
        {"substitute": "greek yogurt", "ratio": "1/2 cup yogurt for 1 cup butter", "context": "Moist baked goods"},
    ],
    "milk": [
        {"substitute": "almond milk", "ratio": "1:1", "context": "General cooking and baking"},
        {"substitute": "oat milk", "ratio": "1:1", "context": "Baking, coffee, sauces"},
        {"substitute": "soy milk", "ratio": "1:1", "context": "Baking and savory dishes"},
        {"substitute": "coconut milk", "ratio": "1:1", "context": "Curries, rich soups, desserts"},
        {"substitute": "water + butter", "ratio": "1 cup water + 1.5 tsp butter for 1 cup whole milk", "context": "Emergency cooking substitute"},
    ],
    "heavy cream": [
        {"substitute": "milk + butter", "ratio": "3/4 cup milk + 1/4 cup melted butter", "context": "Cooking & baking (cannot whip)"},
        {"substitute": "coconut cream", "ratio": "1:1", "context": "Dairy-free cooking, soups, curries, whipping"},
        {"substitute": "greek yogurt + milk", "ratio": "1:1 mixture", "context": "Thickening sauces & dressings"},
        {"substitute": "evaporated milk", "ratio": "1:1", "context": "Cream soups and casseroles"},
    ],
    "buttermilk": [
        {"substitute": "milk + lemon juice", "ratio": "1 cup milk + 1 tbsp lemon juice (rest 5m)", "context": "Pancakes, waffles, biscuits"},
        {"substitute": "milk + white vinegar", "ratio": "1 cup milk + 1 tbsp vinegar (rest 5m)", "context": "General baking"},
        {"substitute": "plain yogurt + milk", "ratio": "3/4 cup yogurt + 1/4 cup milk", "context": "Marinades & baking"},
    ],
    "sour cream": [
        {"substitute": "plain greek yogurt", "ratio": "1:1", "context": "Baking, dips, dollops"},
        {"substitute": "cottage cheese (blended)", "ratio": "1 cup blended cottage cheese + 1 tbsp lemon juice", "context": "Low-fat dips and dressings"},
        {"substitute": "cream cheese + milk", "ratio": "3/4 cup cream cheese + 3 tbsp milk", "context": "Rich sauces and baking"},
    ],
    "egg": [
        {"substitute": "flax egg", "ratio": "1 tbsp ground flaxseed + 3 tbsp water per egg", "context": "Dense baked goods (brownies, pancakes)"},
        {"substitute": "chia egg", "ratio": "1 tbsp chia seeds + 3 tbsp water per egg", "context": "Baking & binding"},
        {"substitute": "applesauce", "ratio": "1/4 cup unsweetened applesauce per egg", "context": "Moist cakes & quick breads"},
        {"substitute": "mashed banana", "ratio": "1/2 ripe banana per egg", "context": "Pancakes, muffins, sweet breads"},
        {"substitute": "silken tofu", "ratio": "1/4 cup puréed silken tofu per egg", "context": "Quiches, dense cakes"},
        {"substitute": "aquafaba", "ratio": "3 tbsp chickpea liquid per egg", "context": "Egg whites & light baking"},
    ],
    "all-purpose flour": [
        {"substitute": "whole wheat flour", "ratio": "1:1 (add 1-2 tbsp extra liquid)", "context": "Rustic breads and baked goods"},
        {"substitute": "oat flour", "ratio": "1:1 by weight (or 1 1/3 cup oat flour for 1 cup AP flour)", "context": "Pancakes, cookies, quick breads"},
        {"substitute": "almond flour", "ratio": "1:1 (may need extra egg for binding)", "context": "Gluten-free / keto baking"},
        {"substitute": "gluten-free 1-to-1 baking blend", "ratio": "1:1", "context": "Universal gluten-free baking"},
    ],
    "bread crumbs": [
        {"substitute": "crushed crackers", "ratio": "1:1", "context": "Meatballs, meatloaf, toppings"},
        {"substitute": "rolled oats (pulsed)", "ratio": "1:1", "context": "Binding in patties and meatloaf"},
        {"substitute": "crushed cornflakes / cereal", "ratio": "1:1", "context": "Crispy fried or baked coating"},
        {"substitute": "panko", "ratio": "1:1", "context": "Extra crispy coating"},
    ],
    "granulated sugar": [
        {"substitute": "brown sugar", "ratio": "1:1", "context": "Baking with richer caramel note"},
        {"substitute": "honey", "ratio": "3/4 cup honey for 1 cup sugar (reduce liquid by 2 tbsp)", "context": "Baking & sweetening"},
        {"substitute": "maple syrup", "ratio": "3/4 cup syrup for 1 cup sugar (reduce liquid by 3 tbsp)", "context": "Sweetening & glazing"},
        {"substitute": "erythritol / monkfruit", "ratio": "1:1", "context": "Keto / sugar-free baking"},
    ],
    "brown sugar": [
        {"substitute": "white sugar + molasses", "ratio": "1 cup white sugar + 1 tbsp molasses", "context": "Exact flavor match"},
        {"substitute": "white sugar + maple syrup", "ratio": "1 cup white sugar + 1 tbsp maple syrup", "context": "Baking substitute"},
        {"substitute": "coconut sugar", "ratio": "1:1", "context": "Warm caramel flavor"},
    ],
    "soy sauce": [
        {"substitute": "tamari", "ratio": "1:1", "context": "Gluten-free 1:1 substitute"},
        {"substitute": "coconut aminos", "ratio": "1:1", "context": "Soy-free & low-sodium alternative"},
        {"substitute": "worcestershire sauce + water", "ratio": "1:1 mix", "context": "Savory marinades & stir-fry"},
    ],
    "mayonnaise": [
        {"substitute": "greek yogurt", "ratio": "1:1", "context": "Salad dressings, cold pasta salads"},
        {"substitute": "mashed avocado", "ratio": "1:1", "context": "Sandwich spreads and wraps"},
        {"substitute": "hummus", "ratio": "1:1", "context": "Flavorful sandwich spread"},
        {"substitute": "sour cream", "ratio": "1:1", "context": "Creamy dips and slaws"},
    ],
    "lemon juice": [
        {"substitute": "lime juice", "ratio": "1:1", "context": "Direct citrus replacement"},
        {"substitute": "white wine vinegar", "ratio": "1/2 tbsp vinegar for 1 tbsp lemon juice", "context": "Salad dressings and deglazing"},
        {"substitute": "apple cider vinegar", "ratio": "1/2 tbsp ACV for 1 tbsp lemon juice", "context": "Marinades and savory dishes"},
    ],
    "white wine": [
        {"substitute": "chicken / vegetable broth + 1 tsp lemon juice", "ratio": "1:1", "context": "Deglazing and cooking sauces"},
        {"substitute": "apple cider vinegar + water", "ratio": "1 part vinegar to 1 part water", "context": "Acidic cooking balance"},
    ],
    "red wine": [
        {"substitute": "beef broth + 1 tbsp red wine vinegar", "ratio": "1:1", "context": "Braising and rich stews"},
        {"substitute": "pomegranate juice or cranberry juice (unsweetened)", "ratio": "1:1", "context": "Sauces & marinades"},
    ],
    "garlic": [
        {"substitute": "garlic powder", "ratio": "1/8 tsp powder per 1 clove fresh garlic", "context": "Seasoning blends and sauces"},
        {"substitute": "shallot / onion minced", "ratio": "1 tbsp minced shallot per 1 clove garlic", "context": "Aromatics for sauté"},
    ],
    "onion": [
        {"substitute": "onion powder", "ratio": "1 tbsp powder for 1 medium onion", "context": "Sauces, rubs, and soups"},
        {"substitute": "shallots", "ratio": "3 shallots per 1 medium onion", "context": "Delicate sauté and dressings"},
        {"substitute": "leeks / green onions", "ratio": "1:1 volume", "context": "Soups, stir-fries, and stews"},
    ],
    "cornstarch": [
        {"substitute": "all-purpose flour", "ratio": "2 tbsp flour for 1 tbsp cornstarch", "context": "Sauce and gravy thickening"},
        {"substitute": "arrowroot powder", "ratio": "1:1", "context": "Clear glossy sauces and fruit fillings"},
        {"substitute": "tapioca starch", "ratio": "2 tbsp tapioca for 1 tbsp cornstarch", "context": "Pies and high-acid fillings"},
    ],
    "tomato paste": [
        {"substitute": "tomato sauce (simmered down)", "ratio": "3 tbsp tomato sauce cooked down to 1 tbsp", "context": "Stews, soups, pastas"},
        {"substitute": "ketchup", "ratio": "1:1 (reduce other sugars in recipe)", "context": "Quick savory cooking"},
    ],
    "parmesan cheese": [
        {"substitute": "pecorino romano", "ratio": "1:1 (slightly saltier)", "context": "Pastas, salads, toppings"},
        {"substitute": "asiago / grana padano", "ratio": "1:1", "context": "Gratins, sauces, garnishes"},
        {"substitute": "nutritional yeast", "ratio": "1:1", "context": "Vegan cheese replacement"},
    ],
    "cheddar cheese": [
        {"substitute": "colby / monterey jack", "ratio": "1:1", "context": "Melting, burgers, casseroles"},
        {"substitute": "gouda", "ratio": "1:1", "context": "Sandwiches and pasta bakes"},
    ],
    "ricotta cheese": [
        {"substitute": "cottage cheese (drained)", "ratio": "1:1", "context": "Lasagna and stuffed shells"},
        {"substitute": "mascarpone / cream cheese", "ratio": "1:1", "context": "Desserts and rich fillings"},
    ],
    "ground beef": [
        {"substitute": "ground turkey", "ratio": "1:1", "context": "Tacos, pasta sauces, meatballs"},
        {"substitute": "ground pork / chicken", "ratio": "1:1", "context": "Meatloaf, burgers, stir-fries"},
        {"substitute": "lentils / black beans", "ratio": "1 cup cooked lentils for 1/2 lb beef", "context": "Vegetarian chili, pasta, tacos"},
    ],
    "chicken breast": [
        {"substitute": "chicken thighs", "ratio": "1:1 (juicier, cook slightly longer)", "context": "Curries, stir-fries, roasting"},
        {"substitute": "turkey cutlets", "ratio": "1:1", "context": "Pan-frying and baking"},
        {"substitute": "tofu (firm/extra firm)", "ratio": "1:1 pressed tofu", "context": "Vegetarian stir-fry and curries"},
    ],
    "baking powder": [
        {"substitute": "baking soda + cream of tartar", "ratio": "1/4 tsp baking soda + 1/2 tsp cream of tartar for 1 tsp baking powder", "context": "Standard leavening"},
        {"substitute": "baking soda + buttermilk / yogurt", "ratio": "1/4 tsp baking soda + 1/2 cup buttermilk (reduce other liquid)", "context": "Pancakes and cakes"},
    ],
    "baking soda": [
        {"substitute": "baking powder", "ratio": "3 tsp baking powder for 1 tsp baking soda (reduce salt)", "context": "Leavened baking"},
    ],
    "olive oil": [
        {"substitute": "canola oil / vegetable oil", "ratio": "1:1", "context": "High-heat cooking and neutral baking"},
        {"substitute": "avocado oil", "ratio": "1:1", "context": "Sautéing and salad dressings"},
        {"substitute": "melted butter", "ratio": "1:1", "context": "Basting and savory cooking"},
    ],
    "sesame oil": [
        {"substitute": "toasted sesame seeds in neutral oil", "ratio": "1:1", "context": "Asian stir-fries and dressings"},
        {"substitute": "peanut oil", "ratio": "1:1", "context": "High-heat stir fry flavor"},
    ],
    "fresh herbs": [
        {"substitute": "dried equivalent", "ratio": "1 tsp dried for 1 tbsp fresh (1:3 ratio)", "context": "General cooked dishes"},
    ],
}


def normalize_ingredient_key(raw_name: str) -> str:
    """Cleans up ingredient query to match substitution database keys."""
    text = raw_name.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return text


def get_substitutions_for_ingredient(ingredient_name: str) -> list[dict[str, str]]:
    """Retrieves all possible substitutes for a given ingredient."""
    norm = normalize_ingredient_key(ingredient_name)

    # 1. Direct match
    if norm in SUBSTITUTION_KNOWLEDGE_BASE:
        return SUBSTITUTION_KNOWLEDGE_BASE[norm]

    # 2. Key contained in name or vice versa
    for key, subs in SUBSTITUTION_KNOWLEDGE_BASE.items():
        if key in norm or norm in key:
            return subs

    # 3. Word token overlap
    tokens = set(norm.split())
    for key, subs in SUBSTITUTION_KNOWLEDGE_BASE.items():
        key_tokens = set(key.split())
        if tokens & key_tokens:
            return subs

    return []


def suggest_recipe_substitutions(
    missing_ingredients: list[str],
    available_pantry: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Analyzes missing recipe ingredients and returns tailored substitute recommendations.

    If available_pantry is provided, it prioritizes substitutes that the user already has in stock.
    """
    suggestions: list[dict[str, Any]] = []
    pantry_set = {normalize_ingredient_key(p) for p in (available_pantry or [])}

    for missing in missing_ingredients:
        subs = get_substitutions_for_ingredient(missing)
        if not subs:
            continue

        scored_subs = []
        for sub in subs:
            sub_name = normalize_ingredient_key(sub["substitute"])
            in_pantry = any(p in sub_name or sub_name in p for p in pantry_set)
            scored_subs.append({
                **sub,
                "in_pantry": in_pantry,
            })

        # Sort so in_pantry items show up first
        scored_subs.sort(key=lambda s: s["in_pantry"], reverse=True)

        suggestions.append({
            "missing_ingredient": missing,
            "substitutes": scored_subs,
            "has_pantry_match": any(s["in_pantry"] for s in scored_subs),
        })

    return suggestions
