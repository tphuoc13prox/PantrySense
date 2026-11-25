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


@pytest.mark.parametrize(
    "title, vegetarian, vegan",
    [
        # Food.com rows that list no meat at all.
        ("Coleman Prime Rib Roast", False, False),
        ("Grilled Wild Boar Chops", False, False),
        ("Pork Tenderloin With Mustard, Lemon and Rosemary", False, False),
        ("Chicken-Vegetable Manicotti With Spinach Sauce", False, False),
        ("Chesapeake Crab Quiche", False, False),
        ("Pot Roast", False, False),
        ("Shepherd's Pie", False, False),
        ("Salmon With Vegan Pesto", False, False),
        ("Vegetarian Lasagna With Chicken", False, False),
        ("Veggie Beef Soup", False, False),
        ("Lentil Sausage Soup", False, False),
        ("Potato Bacon Soup", False, False),
        # "Mock chicken legs" are veal or pork.
        ("Mock Chicken Legs", False, False),
        # Words that only look like meat, and plant stand-ins.
        ("Eggplant Parmesan", True, False),
        ("Veggie Burger", True, True),
        ("Black Bean Burgers", True, True),
        ("Vegan Chorizo Tacos", True, True),
        ("Cauliflower Steak With Chimichurri", True, True),
        ("Portobello Mushroom Steaks", True, True),
        ("Tempeh Bacon", True, True),
        ("Vegetarian Swedish Meatballs", True, True),
        ("Lentil Shepherd's Pie", True, True),
        ("Roast Potatoes", True, True),
        ("Heart Healthy Salad", True, True),
        ("Angel Wings", True, True),
        ("Chicken of the Woods Saute", True, True),
        ("Pate a Choux Puffs", True, True),
        # What the title says about dairy and eggs.
        ("Vegan Mac and Cheese", True, True),
        ("&quot;cheese&quot; Sauce (Dairy Free)", True, True),
        ("A Dilly Potato Salad, No Mayo", True, True),
        ("Thanksgiving Pumpkin Chiffon Pie (Gelatin Free)", True, True),
        ("Hot Chocolate", True, True),
        ("Peach Butter", True, True),
        ("Goat Cheese Tart", True, False),
        ("Honey Glazed Carrots", True, False),
    ],
)
def test_title_adds_what_the_ingredients_miss(title, vegetarian, vegan):
    tags = classify_recipe_dietary(["olive oil", "garlic"], title=title)
    assert (tags["vegetarian"], tags["vegan"]) == (vegetarian, vegan)


def test_title_never_removes_ingredient_flags():
    tags = classify_recipe_dietary(["chicken breast", "butter"], title="Vegan Garden Salad")
    assert tags["vegetarian"] is False and tags["vegan"] is False


def test_source_tags():
    base = ["olive oil", "lemon juice", "rosemary"]
    assert classify_recipe_dietary(base, tags=["Meat", "Roast Beef"])["vegetarian"] is False
    assert classify_recipe_dietary(base, tags=["Easy", "< 60 Mins", "European"])["vegan"] is True
    assert detect_recipe_allergens(base, tags=["Crab"]) == ["shellfish"]
    assert detect_recipe_allergens(base, tags=["Tilapia"]) == ["fish"]
    cheese = classify_recipe_dietary(base, tags=["Cheese"])
    assert cheese["vegetarian"] is True and cheese["vegan"] is False
    # A tofu cheesecake tagged "Cheese" stays vegan; meat tags are never overruled.
    assert classify_recipe_dietary(base, title="Vegan Cheesecake", tags=["Cheesecake"])["vegan"] is True
    assert classify_recipe_dietary(base, title="Vegan Pot Pie", tags=["Poultry"])["vegetarian"] is False


def test_title_allergens_are_animal_only():
    assert detect_recipe_allergens(["water"], title="Miso Salmon &amp; Spinach") == ["fish"]
    assert detect_recipe_allergens(["water"], title="Peanut Butter Cookies") == []
    detailed = detect_recipe_allergens_detailed(["water"], title="Chesapeake Crab Quiche")
    assert detailed["shellfish"]["sources"] == ["title: Chesapeake Crab Quiche"]


def test_breaded_and_offal_products():
    assert classify_recipe_dietary(["fish sticks"])["vegetarian"] is False
    assert "fish" in detect_recipe_allergens(["fish sticks"])
    assert classify_recipe_dietary(["chicken nuggets"])["vegetarian"] is False
    assert classify_recipe_dietary(["sweetbreads"])["vegetarian"] is False


def test_dish_labels_on_fixture():
    data = json.loads((FIXTURES / "allergen_recipes.json").read_text(encoding="utf-8"))
    unsafe = {"vegetarian": 0, "vegan": 0}
    correct = {"vegetarian": 0, "vegan": 0}
    for recipe in data["recipes"]:
        tags = recipe["keywords"] + ([recipe["source_category"]] if recipe["source_category"] else [])
        pred = classify_recipe_dietary(recipe["ingredients"] + recipe["extra_ingredients"],
                                       title=recipe["title"], tags=tags)
        for tag in unsafe:
            gold = recipe[f"dish_{tag}"]
            unsafe[tag] += pred[tag] and not gold
            correct[tag] += pred[tag] == gold
    assert unsafe == {"vegetarian": 0, "vegan": 0}
    assert correct["vegetarian"] >= 96 and correct["vegan"] >= 99
