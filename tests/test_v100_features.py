from __future__ import annotations

import datetime
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.database import pantry_db
from app.backend.main import app
from app.backend.recipes import dietary, nutrition, substitution
from app.backend.recipes.schemas import DietaryFilters, RecipeSearchRequest
from app.backend.recipes.service import RecipeSearchService


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_pantry.db"
    from app.backend.database.init_db import initialize_database
    initialize_database(db_file)
    pantry_db.ensure_pantry_tables(db_file)
    return db_file


def test_dietary_classification():
    veg_ings = ["tomato", "basil", "olive oil", "garlic", "salt"]
    res = dietary.classify_recipe_dietary(veg_ings)
    assert res["vegetarian"] is True
    assert res["vegan"] is True
    assert res["gluten_free"] is True
    assert res["dairy_free"] is True
    assert res["nut_free"] is True

    meat_ings = ["chicken breast", "butter", "flour", "onion"]
    res_meat = dietary.classify_recipe_dietary(meat_ings)
    assert res_meat["vegetarian"] is False
    assert res_meat["vegan"] is False
    assert res_meat["gluten_free"] is False
    assert res_meat["dairy_free"] is False


def test_allergen_detection():
    ings = ["peanut butter", "milk", "all-purpose flour", "shrimp"]
    allergens = dietary.detect_recipe_allergens(ings)
    assert "peanuts" in allergens
    assert "dairy" in allergens
    assert "gluten" in allergens
    assert "shellfish" in allergens


def test_substitution_engine():
    subs = substitution.get_substitutions_for_ingredient("butter")
    assert len(subs) > 0
    sub_names = [s["substitute"] for s in subs]
    assert any("oil" in s for s in sub_names)

    missing = ["butter", "heavy cream"]
    pantry = ["olive oil"]
    suggestions = substitution.suggest_recipe_substitutions(missing, available_pantry=pantry)
    assert len(suggestions) == 2
    # Check that olive oil has in_pantry True
    butter_sub = next(s for s in suggestions if s["missing_ingredient"] == "butter")
    assert butter_sub["has_pantry_match"] is True
    first_opt = butter_sub["substitutes"][0]
    assert first_opt["in_pantry"] is True


def test_nutrition_estimation():
    ings = ["chicken breast", "rice", "broccoli", "olive oil"]
    nut = nutrition.calculate_recipe_nutrition(ings, servings=2)
    assert nut["servings"] == 2
    assert nut["per_serving"]["calories"] > 100
    assert nut["per_serving"]["protein"] > 10
    assert "breakdown" in nut
    assert len(nut["breakdown"]) == 4


def test_pantry_db_crud(temp_db: Path):
    today = datetime.date.today()
    exp_soon = (today + datetime.timedelta(days=2)).isoformat()
    exp_past = (today - datetime.timedelta(days=1)).isoformat()
    fresh = (today + datetime.timedelta(days=10)).isoformat()

    item1 = pantry_db.add_pantry_item("Milk", 1.0, "bottle", "Dairy", exp_soon, database_path=temp_db)
    assert item1["status"] == "expiring_soon"

    item2 = pantry_db.add_pantry_item("Eggs", 12.0, "pcs", "Dairy", exp_past, database_path=temp_db)
    assert item2["status"] == "expired"

    item3 = pantry_db.add_pantry_item("Rice", 2.0, "kg", "Grains", fresh, database_path=temp_db)
    assert item3["status"] == "fresh"

    all_items = pantry_db.get_pantry_items(database_path=temp_db)
    assert len(all_items) == 3

    expiring = pantry_db.get_expiring_pantry_items(days_threshold=3, database_path=temp_db)
    assert len(expiring) == 2

    # Update item
    updated = pantry_db.update_pantry_item(item1["id"], quantity=2.0, database_path=temp_db)
    assert updated["quantity"] == 2.0

    # Delete item
    deleted = pantry_db.delete_pantry_item(item3["id"], database_path=temp_db)
    assert deleted is True
    assert len(pantry_db.get_pantry_items(database_path=temp_db)) == 2


def test_favorites_and_meal_plans(temp_db: Path):
    # Favorites
    fav = pantry_db.add_favorite(1, "Tomato Pasta", {"servings": 2}, database_path=temp_db)
    assert fav["is_favorite"] is True
    assert pantry_db.is_recipe_favorite(1, database_path=temp_db) is True
    assert len(pantry_db.get_favorites(database_path=temp_db)) == 1

    pantry_db.remove_favorite(1, database_path=temp_db)
    assert pantry_db.is_recipe_favorite(1, database_path=temp_db) is False

    # Meal Plans
    plan = pantry_db.set_meal_plan_slot(
        day_of_week="Monday",
        meal_slot="dinner",
        recipe_id=1,
        recipe_title="Tomato Pasta",
        servings=2,
        database_path=temp_db,
    )
    assert plan["day_of_week"] == "Monday"
    assert plan["meal_slot"] == "dinner"

    weekly = pantry_db.get_weekly_meal_plan(database_path=temp_db)
    assert len(weekly) == 1

    # Shopping list
    shop = pantry_db.generate_shopping_list(subtract_pantry=False, database_path=temp_db)
    assert isinstance(shop, list)


def test_api_endpoints():
    client = TestClient(app)

    # Health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    # Pantry endpoints
    pantry_res = client.post("/api/pantry/items", json={"name": "Olive Oil", "quantity": 1, "unit": "bottle"})
    assert pantry_res.status_code == 200
    item_id = pantry_res.json()["id"]

    list_res = client.get("/api/pantry/items")
    assert list_res.status_code == 200
    assert any(i["id"] == item_id for i in list_res.json())

    # Assistant parse steps
    assist_res = client.post(
        "/api/assistant/parse-steps",
        json={"instructions": ["Chop onions.", "Simmer for 15 minutes.", "Bake for 1 hour."]},
    )
    assert assist_res.status_code == 200
    steps = assist_res.json()
    assert len(steps) == 3
    assert steps[1]["timer_minutes"] == 15
    assert steps[2]["timer_minutes"] == 60

    # Assistant substitutions
    sub_res = client.get("/api/assistant/substitutions?ingredient=milk")
    assert sub_res.status_code == 200
    assert len(sub_res.json()["substitutes"]) > 0

    # Cleanup pantry item
    client.delete(f"/api/pantry/items/{item_id}")


def test_clean_and_segment_instructions():
    from app.backend.api.assistant import clean_and_segment_instructions, extract_timer_from_instruction

    # Fragmented sentences with stray commas and quotes
    fragmented = [
        'Cook mushrooms in 2 tbsp butter.',
        'Place chicken between sheets of wax paper; flatten to 1/8\\',
        ',',
        ',',
        'baking dish, overlapping edges.',
        'Bake for 1 hour 30 minutes.',
        'Let cool 1/2 hour before serving.',
    ]
    cleaned = clean_and_segment_instructions(fragmented)
    assert len(cleaned) == 4
    assert 'flatten to 1/8" baking dish' in cleaned[1] or 'flatten to 1/8' in cleaned[1]
    assert ',' not in cleaned

    # Timer extraction tests
    mins1, label1 = extract_timer_from_instruction("Simmer for 15 minutes")
    assert mins1 == 15

    mins2, label2 = extract_timer_from_instruction("Bake for 1 hour 30 minutes")
    assert mins2 == 90

    mins3, label3 = extract_timer_from_instruction("Chill for 1/2 hour")
    assert mins3 == 30

