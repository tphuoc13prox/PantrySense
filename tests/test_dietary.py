from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backend.recipes.dietary import (
    analyze_ingredient,
    classify_recipe_dietary,
    detect_recipe_allergens,
    detect_recipe_allergens_detailed,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "ingredient, expected",
    [
        ("coconut milk", set()),
        ("unsweetened almond milk", {"tree_nuts"}),
        ("soy milk", {"soy"}),
        ("gluten-free flour", set()),
        ("gluten-free all-purpose flour", set()),
        ("vegan butter", set()),
        ("dairy-free margarine", {"soy"}),
        ("cream of tartar", set()),
        ("eggplant", set()),
        ("ground nutmeg", set()),
        ("butternut squash", set()),
        ("water chestnut", set()),
        ("peanut butter", {"peanuts"}),
        ("worcestershire sauce", {"fish", "gluten"}),
        ("basil pesto", {"tree_nuts", "dairy"}),
        ("fish sauce", {"fish"}),
        ("oyster sauce", {"shellfish"}),
        ("ghee", {"dairy"}),
        ("nougat", {"tree_nuts", "eggs", "peanuts"}),
        ("marzipan", {"tree_nuts"}),
        ("tahini", {"sesame"}),
        ("soy sauce", {"soy", "gluten"}),
        ("mayonnaise", {"eggs"}),
        ("lactose-free milk", {"dairy"}),
        ("non-dairy creamer", {"dairy", "soy"}),
        ("rolled oats", {"gluten"}),
        ("corn tortillas", set()),
        ("flour tortillas", {"gluten"}),
        ("oyster mushrooms", set()),
    ],
)
def test_ingredient_allergens(ingredient, expected):
    flags = analyze_ingredient(ingredient)
    assert flags.contains | flags.may_contain == expected


@pytest.mark.parametrize(
    "ingredient, vegetarian, vegan",
    [
        ("worcestershire sauce", False, False),
        ("chicken broth", False, False),
        ("vegetable broth", True, True),
        ("honey", True, False),
        ("gelatin", False, False),
        ("kidney beans", True, True),
        ("hamburger buns", True, True),
        ("celery ribs", True, True),
        ("vegetarian worcestershire sauce", True, True),
        ("butter", True, False),
    ],
)
def test_vegetarian_and_vegan(ingredient, vegetarian, vegan):
    tags = classify_recipe_dietary([ingredient])
    assert tags["vegetarian"] is vegetarian
    assert tags["vegan"] is vegan


def test_may_contain_is_reported_separately():
    detailed = detect_recipe_allergens_detailed(["rolled oats", "milk"])
    assert detailed["gluten"]["level"] == "may_contain"
    assert detailed["dairy"] == {"level": "contains", "sources": ["milk"]}
    assert detect_recipe_allergens(["rolled oats", "milk"]) == ["dairy", "gluten"]


def test_labelled_recipes_recall():
    data = json.loads((FIXTURES / "allergen_recipes.json").read_text(encoding="utf-8"))
    tp = fn = 0
    for recipe in data["recipes"]:
        gold = set(recipe["allergens_contains"]) | set(recipe["allergens_may_contain"])
        found = set(detect_recipe_allergens(recipe["ingredients"]))
        tp += len(gold & found)
        fn += len(gold - found)
    assert tp / (tp + fn) >= 0.97
