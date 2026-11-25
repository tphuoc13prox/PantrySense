from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from app.backend.recipes.ingredient_parser import parse_quantity
from app.backend.recipes.nutrition import calculate_recipe_nutrition, match_food

FIXTURES = Path(__file__).parent / "fixtures"


def test_quantity_changes_calories():
    two_cups = calculate_recipe_nutrition(["2 cups all-purpose flour"], servings=1)
    one_tsp = calculate_recipe_nutrition(["1 tsp all-purpose flour"], servings=1)
    # 2 cups ~ 250 g ~ 910 kcal; 1 tsp ~ 2.6 g ~ 10 kcal
    assert 800 < two_cups["total"]["calories"] < 1000
    assert one_tsp["total"]["calories"] < 15


def test_count_units_use_piece_weight():
    eggs = calculate_recipe_nutrition(["3 large eggs"], servings=1)
    assert eggs["breakdown"][0]["grams"] == pytest.approx(150, abs=1)
    assert eggs["total"]["calories"] == pytest.approx(214.5, abs=1)


def test_package_size_in_parentheses():
    r = calculate_recipe_nutrition(["1 (14.5 ounce) can diced tomatoes"], servings=1)
    assert r["breakdown"][0]["grams"] == pytest.approx(14.5 * 28.35, abs=1)


def test_unknown_ingredient_is_not_given_a_default_profile():
    r = calculate_recipe_nutrition(["2 cups flour", "1 tsp unobtainium dust"], servings=1)
    unknown = r["breakdown"][1]
    assert unknown["status"] == "unknown"
    assert unknown["calories"] == 0.0
    assert r["coverage"] == {
        "ratio": 0.5, "matched": 1, "total": 2, "unknown": ["1 tsp unobtainium dust"],
        "assumed_quantity_or_unit": 0,
    }


def test_ingredient_details_with_placeholder_unit():
    details = [
        {"name": "butter", "quantity": 0.5, "unit": "portion"},  # bare 0.5 -> cup
        {"name": "butter", "quantity": 2.0, "unit": "portion"},  # bare 2 -> tbsp
        {"name": "eggs", "quantity": 2.0, "unit": "portion"},
    ]
    r = calculate_recipe_nutrition(["butter", "butter", "eggs"], servings=1, ingredient_details=details)
    grams = [b["grams"] for b in r["breakdown"]]
    assert grams[0] == pytest.approx(113.6, abs=1)
    assert grams[1] == pytest.approx(28.4, abs=1)
    assert grams[2] == pytest.approx(100, abs=1)


def test_large_bare_numbers_are_read_as_grams():
    r = calculate_recipe_nutrition(["plain flour"], servings=1,
                                   ingredient_details=[{"name": "plain flour", "quantity": 125, "unit": "portion"}])
    assert r["breakdown"][0]["grams"] == pytest.approx(125)


def test_specific_alias_beats_generic():
    assert match_food("coconut milk").key == "coconut milk"
    assert match_food("peanut butter").key == "peanut butter"
    assert match_food("chicken broth").key == "broth"
    assert match_food("gluten-free flour").key == "rice flour"
    assert match_food("unobtainium") is None


def test_result_shape_matches_service_model():
    from app.backend.recipes.service import _build_nutrition_model

    raw = calculate_recipe_nutrition(["chicken breast", "rice", "broccoli", "olive oil"], servings=2)
    model = _build_nutrition_model(raw)
    assert model.servings == 2
    assert len(model.breakdown) == 4


def test_reference_error_is_below_baseline():
    """Food.com per-serving calories as reference; the old code had ~72% median error."""
    data = json.loads((FIXTURES / "nutrition_reference.json").read_text(encoding="utf-8"))
    errors = []
    for item in data["aligned"]:
        details = [{"name": n, "quantity": parse_quantity(q), "unit": "portion"}
                   for n, q in zip(item["parts"], item["quantities"], strict=True)]
        r = calculate_recipe_nutrition(item["parts"], servings=item["servings"], ingredient_details=details)
        ref = item["reference_per_serving"]["calories"]
        errors.append(abs(r["per_serving"]["calories"] - ref) / ref)
    assert statistics.median(errors) < 0.45
