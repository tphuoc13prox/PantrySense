from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")
_STRIP_PUNCT_RE = re.compile(r"['\"(),:;]+")
_TRAILING_STOP_RE = re.compile(r"\s+(?:and|with|or|in|of|for|to|the|a|an|from|by|on|at|as|into)$")
_LEADING_MEASURE_RE = re.compile(r"^\d+(?:/\d+|\.\d+)?(?:-inch|-oz|-ounce|-lb|-pound|-gram|-g)?\s*")
_PERCENT_FAT_RE = re.compile(r"^[12]\s+(?:low-fat|fat|reduced-fat|part-skim|skim|evaporated|fat-free)?\s*")

INGREDIENT_SYNONYMS = {
    # Dialect synonyms
    "bell pepper": "pepper",
    "capsicum": "pepper",
    "tomatoes": "tomato",
    "eggs": "egg",
    "egg tomato": "roma tomato",
    "egg tomatoes": "roma tomato",
    "plum tomato": "roma tomato",
    "plum tomatoes": "roma tomato",
    "aubergine": "eggplant",
    "aubergines": "eggplant",
    "courgette": "zucchini",
    "courgettes": "zucchini",
    "spring onion": "green onion",
    "spring onions": "green onion",
    "scallions": "green onion",
    "scallion": "green onion",
    "coriander": "cilantro",
    "bicarbonate of soda": "baking soda",
    "caster sugar": "sugar",
    "icing sugar": "powdered sugar",
    "confectioners sugar": "powdered sugar",
    "powdered 10x sugar": "powdered sugar",
    "kraft 100 parmesan": "parmesan cheese",
    "bacardi 151 rum": "rum",
    # Compound & truncated dataset entries
    "tomatoes and green": "tomato",
    "tomatoes with juice": "canned tomato",
    "tomato with juice": "canned tomato",
    "tomatoes with herb": "canned tomato",
    "tomatoes with jalapeno": "canned tomato",
    "tomatoes with garlic": "tomato",
    "tomatoes with onion": "tomato",
    "tomatoes with basil": "tomato",
    "tomatoes seasoned with": "tomato",
    "diced tomatoes with": "diced tomato",
    "crushed tomatoes in": "crushed tomato",
    "crushed pineapple with": "pineapple",
    "pineapple chunks in": "pineapple",
    "monterey jack and": "monterey jack cheese",
    "mushroom stems and": "mushroom",
    "cream cheese with": "cream cheese",
    "green onions with": "green onion",
    "white tuna in": "tuna",
    "albacore tuna in": "tuna",
    "mandarin oranges in": "mandarin orange",
    "strawberries with sugar": "strawberry",
    "lemon zest of": "lemon zest",
    "lime juice of": "lime juice",
    "lemon rind of": "lemon zest",
    "orange zest of": "orange zest",
    "orange rind of": "orange zest",
    "lime zest of": "lime zest",
    "lemon juice and": "lemon juice",
    "lime juice and": "lime juice",
    "orange juice and": "orange juice",
    "chipotle chiles in": "chipotle chile",
    "chipotle chile in": "chipotle chile",
    "half and half": "half-and-half",
    "cheddar and colby": "cheddar cheese",
    "cheddar and american": "cheddar cheese",
    "parmesan and mozzarella": "parmesan cheese",
    "onion and garlic": "onion",
    "garlic and cheese": "garlic",
    "cabbage and carrot": "cabbage",
}


def normalize_ingredient(value: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", value.strip().lower())
    if not cleaned or "character" in cleaned or cleaned.isdigit():
        return ""

    # Strip quotes, parens, trailing colons/commas
    cleaned = _STRIP_PUNCT_RE.sub("", cleaned).strip()

    if cleaned in INGREDIENT_SYNONYMS:
        return INGREDIENT_SYNONYMS[cleaned]

    # Clean leading measure or fat %
    if _PERCENT_FAT_RE.match(cleaned):
        cleaned = _PERCENT_FAT_RE.sub("", cleaned).strip()
    elif _LEADING_MEASURE_RE.match(cleaned):
        cleaned = _LEADING_MEASURE_RE.sub("", cleaned).strip()

    # Repeatedly strip trailing stopwords (e.g. 'diced tomatoes with' -> 'diced tomatoes')
    while True:
        sub = _TRAILING_STOP_RE.sub("", cleaned).strip()
        if sub == cleaned:
            break
        cleaned = sub

    if cleaned in INGREDIENT_SYNONYMS:
        return INGREDIENT_SYNONYMS[cleaned]

    if cleaned.endswith("ies") and len(cleaned) > 3:
        cleaned = f"{cleaned[:-3]}y"
    elif cleaned.endswith("oes") and len(cleaned) > 3:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
        cleaned = cleaned[:-1]

    return INGREDIENT_SYNONYMS.get(cleaned, cleaned)


def normalize_ingredients(values: list[str]) -> list[str]:
    normalized = {
        ingredient
        for value in values
        if (ingredient := normalize_ingredient(value))
    }
    return sorted(normalized)


@dataclass(frozen=True)
class IngredientMatch:
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    matched_count: int
    required_count: int
    coverage: float


class IngredientMatcher:
    def match(self, user_ingredients: list[str], recipe_ingredients: list[str]) -> IngredientMatch:
        normalized_user = set(normalize_ingredients(user_ingredients))
        normalized_recipe = normalize_ingredients(recipe_ingredients)
        recipe_set = set(normalized_recipe)

        matched = sorted(normalized_user.intersection(recipe_set))
        missing = [ingredient for ingredient in normalized_recipe if ingredient not in normalized_user]
        required_count = len(normalized_recipe)
        coverage = len(matched) / required_count if required_count else 0.0

        return IngredientMatch(
            matched_ingredients=matched,
            missing_ingredients=missing,
            matched_count=len(matched),
            required_count=required_count,
            coverage=coverage,
        )
