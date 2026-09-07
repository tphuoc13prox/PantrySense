from __future__ import annotations

import re
from typing import Any

# Non-vegetarian and non-vegan indicator keywords
MEAT_KEYWORDS = {
    "chicken", "pork", "beef", "turkey", "duck", "lamb", "veal", "bacon",
    "ham", "sausage", "pepperoni", "salami", "prosciutto", "pancetta", "chorizo",
    "lard", "gelatin", "meat", "steak", "rib", "ribs", "venison", "quail",
    "anchovy", "anchovies", "fish", "salmon", "tuna", "cod", "tilapia", "trout",
    "halibut", "snapper", "shrimp", "prawn", "prawns", "crab", "lobster", "clam",
    "clams", "mussel", "mussels", "oyster", "oysters", "scallop", "scallops",
    "squid", "calamari", "octopus", "seafood",
}

SEAFOOD_KEYWORDS = {
    "anchovy", "anchovies", "fish", "salmon", "tuna", "cod", "tilapia", "trout",
    "halibut", "snapper", "shrimp", "prawn", "prawns", "crab", "lobster", "clam",
    "clams", "mussel", "mussels", "oyster", "oysters", "scallop", "scallops",
    "squid", "calamari", "octopus", "seafood",
}

DAIRY_KEYWORDS = {
    "milk", "butter", "cheese", "cream", "yogurt", "ghee", "casein", "whey",
    "half-and-half", "custard", "buttermilk", "parmesan", "cheddar", "mozzarella",
    "ricotta", "feta", "gouda", "provolone", "swiss cheese", "sour cream",
    "heavy cream", "condensed milk", "evaporated milk", "cream cheese",
}

EGG_KEYWORDS = {
    "egg", "eggs", "egg yolk", "egg white", "mayonnaise", "meringue",
}

GLUTEN_KEYWORDS = {
    "flour", "wheat", "all-purpose flour", "bread flour", "pasta", "noodle",
    "spaghetti", "macaroni", "fettuccine", "barley", "rye", "couscous", "semolina",
    "bread", "breadcrumbs", "tortilla", "soy sauce", "graham", "cracker",
    "malt", "beer", "orzo",
}

NUT_KEYWORDS = {
    "peanut", "peanuts", "peanut butter", "almond", "almonds", "walnut", "walnuts",
    "pecan", "pecans", "cashew", "cashews", "hazelnut", "hazelnuts", "pistachio",
    "pistachios", "macadamia", "pine nut", "pine nuts", "nut", "nuts",
}

SOY_KEYWORDS = {
    "soy", "soy sauce", "tofu", "edamame", "miso", "tempeh", "soy milk", "tamari",
}

HIGH_CARB_KEYWORDS = {
    "sugar", "flour", "rice", "pasta", "noodle", "potato", "potatoes", "bread",
    "honey", "syrup", "corn", "cornstarch", "oats", "cereal", "banana", "sweet potato",
}


def classify_recipe_dietary(ingredients: list[str]) -> dict[str, bool]:
    """Evaluates an ingredient list against dietary criteria."""
    normalized_list = [ing.lower().strip() for ing in ingredients if ing]
    all_tokens = " ".join(normalized_list)

    has_meat = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in MEAT_KEYWORDS)
    has_seafood = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in SEAFOOD_KEYWORDS)
    has_dairy = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in DAIRY_KEYWORDS)
    has_egg = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in EGG_KEYWORDS)
    has_gluten = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in GLUTEN_KEYWORDS)
    has_nuts = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in NUT_KEYWORDS)
    has_high_carb = any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in HIGH_CARB_KEYWORDS)

    is_vegetarian = not has_meat and not has_seafood
    is_vegan = is_vegetarian and not has_dairy and not has_egg and "honey" not in all_tokens
    is_gluten_free = not has_gluten
    is_dairy_free = not has_dairy
    is_nut_free = not has_nuts
    is_keto_low_carb = not has_high_carb

    return {
        "vegetarian": is_vegetarian,
        "vegan": is_vegan,
        "gluten_free": is_gluten_free,
        "dairy_free": is_dairy_free,
        "nut_free": is_nut_free,
        "keto_low_carb": is_keto_low_carb,
    }


def detect_recipe_allergens(ingredients: list[str]) -> list[str]:
    """Identifies common food allergens present in the ingredient list."""
    normalized_list = [ing.lower().strip() for ing in ingredients if ing]
    all_tokens = " ".join(normalized_list)
    allergens: list[str] = []

    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in ("peanut", "peanuts", "peanut butter")):
        allergens.append("peanuts")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in ("almond", "walnut", "pecan", "cashew", "hazelnut", "pistachio", "macadamia")):
        allergens.append("tree_nuts")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in DAIRY_KEYWORDS):
        allergens.append("dairy")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in EGG_KEYWORDS):
        allergens.append("eggs")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in GLUTEN_KEYWORDS):
        allergens.append("gluten")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in SOY_KEYWORDS):
        allergens.append("soy")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in ("shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster", "scallop", "squid", "calamari", "octopus")):
        allergens.append("shellfish")
    if any(re.search(rf"\b{re.escape(k)}\b", all_tokens) for k in ("fish", "salmon", "tuna", "cod", "tilapia", "trout", "halibut", "snapper", "anchovy")):
        allergens.append("fish")

    return sorted(list(set(allergens)))
