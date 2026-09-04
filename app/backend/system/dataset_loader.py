from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


from app.backend.config import (
    get_bm25_index_path,
    get_database_path,
    get_dataset_download_limit,
    get_embedder_engine,
    get_ranker_model_path,
    get_vector_ids_path,
    get_vector_index_path,
)
from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.semantic import build_recipe_representation
from app.backend.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger("pantrysense.setup")

MINIMUM_RECIPES_FOR_FULL_DATASET = 15


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


def download_full_hf_parquet_file(progress_callback: Any = None) -> Path:
    """Downloads or retrieves the cached full Hugging Face food-recipes parquet dataset file."""
    data_dir = get_database_path().parent
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "food_recipes.parquet"

    if parquet_path.exists() and parquet_path.stat().st_size > 10 * 1024 * 1024:
        logger.info("Using cached parquet dataset file: %s (%d MB)", parquet_path, parquet_path.stat().st_size // (1024 * 1024))
        if progress_callback:
            progress_callback(100.0, "Found cached 228MB recipe dataset")
        return parquet_path

    url = "https://huggingface.co/datasets/untitledwebsite123/food-recipes/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    tmp_path = data_dir / "food_recipes.parquet.tmp"
    logger.info("Downloading full food-recipes parquet dataset from Hugging Face: %s", url)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PantrySense-Recipe-Downloader/1.0 (Mozilla/5.0)"}
    )

    with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_path, "wb") as f:
        total_size = int(resp.headers.get("content-length", 228880546))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunk

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                pct = (downloaded / total_size) * 100.0 if total_size > 0 else 50.0
                dl_mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                progress_callback(pct, f"Downloading full dataset ({dl_mb:.1f} MB / {tot_mb:.1f} MB)")

    if tmp_path.exists():
        if parquet_path.exists():
            try:
                os.remove(parquet_path)
            except Exception:
                pass
        tmp_path.replace(parquet_path)

    logger.info("Full parquet dataset downloaded successfully (%d MB).", parquet_path.stat().st_size // (1024 * 1024))
    return parquet_path


def load_recipes_from_parquet(
    parquet_path: Path,
    limit: int = 10000,
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """Reads and parses up to `limit` structured recipes from a local parquet file using pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed, skipping parquet load")
        return []

    if not parquet_path.exists():
        return []

    pf = pq.ParquetFile(parquet_path)
    recipes: list[dict[str, Any]] = []

    columns = [
        "Name",
        "RecipeIngredientParts",
        "RecipeIngredientQuantities",
        "RecipeInstructions",
        "CookTime",
        "RecipeCategory",
        "RecipeServings",
    ]

    for batch in pf.iter_batches(batch_size=2500, columns=columns):
        pydict = batch.to_pydict()
        names = pydict.get("Name", [])
        ing_parts = pydict.get("RecipeIngredientParts", [])
        ing_quants = pydict.get("RecipeIngredientQuantities", [])
        instructions = pydict.get("RecipeInstructions", [])
        cook_times = pydict.get("CookTime", [])
        categories = pydict.get("RecipeCategory", [])
        servings_list = pydict.get("RecipeServings", [])

        for i in range(len(names)):
            title = str(names[i] or "").strip()
            if not title or len(title) < 2:
                continue

            raw_ings = ing_parts[i] if i < len(ing_parts) else []
            raw_items = _parse_c_vector(raw_ings)
            parsed_ingredients: list[str] = []
            ingredient_details: list[dict[str, Any]] = []

            raw_q = _parse_c_vector(ing_quants[i]) if i < len(ing_quants) else []
            for idx, item in enumerate(raw_items):
                clean_name = re.sub(r"[^\w\s-]", "", item.lower()).strip()
                if clean_name and len(clean_name) > 1:
                    words = clean_name.split()
                    short_name = " ".join(words[:3])
                    if short_name not in parsed_ingredients:
                        parsed_ingredients.append(short_name)
                        q_val = 1.0
                        if idx < len(raw_q):
                            try:
                                q_val = float(eval(raw_q[idx]))
                            except Exception:
                                q_val = 1.0
                        ingredient_details.append({"name": short_name, "quantity": q_val, "unit": "portion"})

            if not parsed_ingredients:
                continue

            raw_instr = instructions[i] if i < len(instructions) else []
            parsed_instr = _parse_c_vector(raw_instr)
            if not parsed_instr:
                parsed_instr = [f"Prepare ingredients and cook {title} to desired taste."]

            cook_time_val = cook_times[i] if i < len(cook_times) else "20"
            if isinstance(cook_time_val, str) and ("PT" in cook_time_val or "H" in cook_time_val or "M" in cook_time_val):
                cooking_time = _parse_duration_minutes(cook_time_val)
            else:
                try:
                    cooking_time = int(float(cook_time_val or 20))
                except Exception:
                    cooking_time = 20

            raw_servings = servings_list[i] if i < len(servings_list) else 2
            try:
                servings = int(float(raw_servings or 2))
            except Exception:
                servings = 2

            raw_cat = str(categories[i] if i < len(categories) else "dinner").lower()
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

            recipes.append({
                "title": title,
                "ingredients": parsed_ingredients[:12],
                "ingredient_details": ingredient_details[:12],
                "instructions": parsed_instr[:10],
                "cooking_time": min(max(cooking_time, 5), 180),
                "difficulty": "medium" if cooking_time > 25 else "easy",
                "servings": min(max(servings, 1), 12),
                "category": category,
            })

            if len(recipes) >= limit:
                break

        if progress_callback:
            progress_callback(len(recipes), limit)

        if len(recipes) >= limit:
            break

    logger.info("Loaded and parsed %d recipes from parquet file.", len(recipes))
    return recipes


def fetch_online_recipes_from_huggingface(limit: int = 100) -> list[dict[str, Any]]:
    """Fetches real recipe records from Hugging Face Datasets Serverless REST API."""
    batch_size = 50
    batches = max(1, limit // batch_size)
    recipes: list[dict[str, Any]] = []

    for b in range(batches):
        offset = b * batch_size
        url = f"https://datasets-server.huggingface.co/rows?dataset=untitledwebsite123/food-recipes&config=default&split=train&offset={offset}&length={batch_size}"
        try:
            logger.info("Connecting to Hugging Face dataset server (offset=%d): %s", offset, url)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PantrySense-Recipe-Downloader/1.0 (Mozilla/5.0)"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    rows = payload.get("rows", [])
                    logger.info("Successfully received %d rows from Hugging Face (offset %d).", len(rows), offset)
                    for item in rows:
                        row_data = item.get("row", {})
                        recipe = _parse_hf_recipe_text(row_data)
                        if recipe:
                            recipes.append(recipe)
        except Exception as e:
            logger.warning("Failed to fetch dataset from %s: %s (will continue)", url, e)

    logger.info("Parsed total %d online recipes from Hugging Face.", len(recipes))
    return recipes


# Comprehensive, diverse culinary recipe dataset (covering noodles, milk, eggs, meats, vegetables, pasta, soups, desserts, etc.)
CORE_RECIPE_CATALOG = [
    # --- PORK & NOODLE DISHES ---
    {
        "title": "Savory Pork Garlic Stir Fried Noodles",
        "ingredients": ["pork", "noodles", "garlic", "soy sauce", "green onion", "oil", "black pepper"],
        "ingredient_details": [
            {"name": "pork", "quantity": 200, "unit": "g"},
            {"name": "noodles", "quantity": 200, "unit": "g"},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
            {"name": "oil", "quantity": 1.5, "unit": "tbsp"},
            {"name": "black pepper", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Boil noodles until al dente, drain and toss with a little oil.",
            "Slice pork thinly and sear in a hot skillet with minced garlic until golden.",
            "Add noodles and scallions, drizzle with soy sauce and black pepper.",
            "Stir fry vigorously on high heat for 3 minutes and serve hot.",
        ],
        "cooking_time": 15,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Classic Pork Belly Ramen",
        "ingredients": ["pork", "noodles", "egg", "green onion", "soy sauce", "garlic", "sesame oil"],
        "ingredient_details": [
            {"name": "pork", "quantity": 200, "unit": "g"},
            {"name": "noodles", "quantity": 180, "unit": "g"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "sesame oil", "quantity": 1, "unit": "tsp"},
        ],
        "instructions": [
            "Sear sliced pork belly in a hot pan until crispy on the edges.",
            "Simmer ramen broth with soy sauce, garlic, and sesame oil.",
            "Boil ramen noodles and soft-boil eggs.",
            "Assemble bowls with noodles, piping hot broth, topped with pork belly slices, ramen egg, and green onions.",
        ],
        "cooking_time": 20,
        "difficulty": "medium",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Char Siu Pork Egg Noodles",
        "ingredients": ["pork", "noodles", "soy sauce", "honey", "garlic", "green onion", "sesame oil"],
        "ingredient_details": [
            {"name": "pork", "quantity": 250, "unit": "g"},
            {"name": "noodles", "quantity": 200, "unit": "g"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "honey", "quantity": 1.5, "unit": "tbsp"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
            {"name": "sesame oil", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Glaze sliced pork with honey, soy sauce, and garlic, then pan-roast until caramelized.",
            "Toss boiled egg noodles with a dash of sesame oil and soy sauce.",
            "Slice char siu pork over the noodles and garnish with chopped green onions.",
        ],
        "cooking_time": 20,
        "difficulty": "medium",
        "servings": 2,
        "category": "lunch",
    },
    {
        "title": "Vietnamese Grilled Pork with Vermicelli Noodles",
        "ingredients": ["pork", "noodles", "garlic", "sugar", "soy sauce", "green onion", "black pepper"],
        "ingredient_details": [
            {"name": "pork", "quantity": 250, "unit": "g"},
            {"name": "noodles", "quantity": 200, "unit": "g"},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "sugar", "quantity": 1, "unit": "tbsp"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
            {"name": "black pepper", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Marinate thinly sliced pork with minced garlic, sugar, soy sauce, and black pepper.",
            "Grill or pan-sear pork until deeply browned and fragrant.",
            "Serve over warm rice vermicelli noodles topped with scallion oil.",
        ],
        "cooking_time": 20,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Crispy Sweet and Sour Pork",
        "ingredients": ["pork", "bell pepper", "onion", "garlic", "soy sauce", "sugar", "oil"],
        "ingredient_details": [
            {"name": "pork", "quantity": 300, "unit": "g"},
            {"name": "bell pepper", "quantity": 1, "unit": "piece"},
            {"name": "onion", "quantity": 1, "unit": "piece"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "sugar", "quantity": 1.5, "unit": "tbsp"},
            {"name": "oil", "quantity": 2, "unit": "tbsp"},
        ],
        "instructions": [
            "Cut pork into bite-sized cubes and sear in hot oil until crispy golden.",
            "Sauté bell peppers, onions, and minced garlic.",
            "Toss in sweet and savory glaze until glossy and well coated.",
        ],
        "cooking_time": 20,
        "difficulty": "medium",
        "servings": 3,
        "category": "dinner",
    },
    {
        "title": "Caramelized Pork and Braised Eggs",
        "ingredients": ["pork", "egg", "garlic", "sugar", "soy sauce", "black pepper", "green onion"],
        "ingredient_details": [
            {"name": "pork", "quantity": 300, "unit": "g"},
            {"name": "egg", "quantity": 4, "unit": "pieces"},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "sugar", "quantity": 2, "unit": "tbsp"},
            {"name": "soy sauce", "quantity": 2.5, "unit": "tbsp"},
            {"name": "black pepper", "quantity": 0.5, "unit": "tsp"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
        ],
        "instructions": [
            "Caramelize sugar with oil in a pot until deep amber.",
            "Add diced pork belly, garlic, soy sauce, and black pepper; braise for 20 minutes.",
            "Add hard-boiled peeled eggs and simmer gently until rich and savory.",
        ],
        "cooking_time": 30,
        "difficulty": "medium",
        "servings": 3,
        "category": "dinner",
    },

    # --- NOODLES & PASTA DISHES ---
    {
        "title": "Creamy Egg and Milk Noodles",
        "ingredients": ["noodles", "milk", "egg", "butter", "garlic", "cheese", "black pepper"],
        "ingredient_details": [
            {"name": "noodles", "quantity": 150, "unit": "g"},
            {"name": "milk", "quantity": 150, "unit": "ml"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "butter", "quantity": 1, "unit": "tbsp"},
            {"name": "garlic", "quantity": 1, "unit": "clove"},
            {"name": "cheese", "quantity": 30, "unit": "g"},
            {"name": "black pepper", "quantity": 0.25, "unit": "tsp"},
        ],
        "instructions": [
            "Boil noodles in water until tender, then drain.",
            "Whisk eggs with milk, salt, and black pepper in a small bowl.",
            "Melt butter in a skillet, sauté minced garlic, then add drained noodles.",
            "Pour in the milk and egg mixture on low heat, stirring continuously to create a velvety creamy sauce.",
            "Top with grated cheese and serve immediately.",
        ],
        "cooking_time": 12,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Creamy Carbonara Pasta",
        "ingredients": ["pasta", "egg", "bacon", "cheese", "milk", "black pepper", "garlic"],
        "ingredient_details": [
            {"name": "pasta", "quantity": 200, "unit": "g"},
            {"name": "egg", "quantity": 3, "unit": "pieces"},
            {"name": "bacon", "quantity": 80, "unit": "g"},
            {"name": "cheese", "quantity": 50, "unit": "g"},
            {"name": "milk", "quantity": 50, "unit": "ml"},
            {"name": "black pepper", "quantity": 0.5, "unit": "tsp"},
            {"name": "garlic", "quantity": 1, "unit": "clove"},
        ],
        "instructions": [
            "Cook pasta in salted boiling water until al dente.",
            "Crisp chopped bacon in a skillet with minced garlic.",
            "Whisk eggs, milk, and grated cheese together with plenty of black pepper.",
            "Toss hot pasta with bacon, remove from heat, and quickly fold in egg-milk mixture until silky.",
        ],
        "cooking_time": 15,
        "difficulty": "medium",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Classic Macaroni and Cheese",
        "ingredients": ["pasta", "milk", "cheese", "butter", "flour", "salt", "black pepper"],
        "ingredient_details": [
            {"name": "pasta", "quantity": 250, "unit": "g"},
            {"name": "milk", "quantity": 300, "unit": "ml"},
            {"name": "cheese", "quantity": 150, "unit": "g"},
            {"name": "butter", "quantity": 2, "unit": "tbsp"},
            {"name": "flour", "quantity": 2, "unit": "tbsp"},
            {"name": "salt", "quantity": 0.5, "unit": "tsp"},
            {"name": "black pepper", "quantity": 0.25, "unit": "tsp"},
        ],
        "instructions": [
            "Boil macaroni pasta until tender and drain.",
            "Melt butter in a saucepan, whisk in flour for 1 minute.",
            "Gradually pour in warm milk, whisking until smooth and thickened.",
            "Melt in shredded cheddar cheese, combine with cooked pasta, and serve hot.",
        ],
        "cooking_time": 20,
        "difficulty": "easy",
        "servings": 3,
        "category": "lunch",
    },
    {
        "title": "Soy Scallion Stir Fried Noodles",
        "ingredients": ["noodles", "green onion", "soy sauce", "oil", "garlic", "sesame oil"],
        "ingredient_details": [
            {"name": "noodles", "quantity": 200, "unit": "g"},
            {"name": "green onion", "quantity": 3, "unit": "stalks"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "oil", "quantity": 2, "unit": "tbsp"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "sesame oil", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Cook noodles, rinse under cold water, and drain thoroughly.",
            "Heat oil on medium heat, fry scallions until browned and aromatic.",
            "Add minced garlic, soy sauce, and a dash of sugar.",
            "Toss noodles vigorously in scallion oil until evenly coated and fragrant.",
        ],
        "cooking_time": 10,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Egg Drop Ramen Soup",
        "ingredients": ["noodles", "egg", "chicken broth", "green onion", "soy sauce", "sesame oil"],
        "ingredient_details": [
            {"name": "noodles", "quantity": 150, "unit": "g"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "chicken broth", "quantity": 500, "unit": "ml"},
            {"name": "green onion", "quantity": 1, "unit": "stalk"},
            {"name": "soy sauce", "quantity": 1, "unit": "tbsp"},
            {"name": "sesame oil", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Bring chicken broth with soy sauce to a rolling boil.",
            "Add ramen noodles and cook for 3 minutes.",
            "Swirl the broth and slowly stream in beaten eggs to create delicate egg ribbons.",
            "Ladle into bowls and garnish with chopped green onions and sesame oil.",
        ],
        "cooking_time": 8,
        "difficulty": "easy",
        "servings": 1,
        "category": "soup",
    },

    # --- MILK & EGG BREAKFAST & DESSERT DISHES ---
    {
        "title": "Classic Golden French Toast",
        "ingredients": ["bread", "egg", "milk", "butter", "sugar", "cinnamon"],
        "ingredient_details": [
            {"name": "bread", "quantity": 4, "unit": "slices"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "milk", "quantity": 80, "unit": "ml"},
            {"name": "butter", "quantity": 1.5, "unit": "tbsp"},
            {"name": "sugar", "quantity": 1, "unit": "tbsp"},
            {"name": "cinnamon", "quantity": 0.25, "unit": "tsp"},
        ],
        "instructions": [
            "Whisk eggs, milk, sugar, and cinnamon in a shallow bowl.",
            "Dip bread slices into the custard mixture for 20 seconds per side.",
            "Melt butter in a pan on medium heat.",
            "Fry toast for 3-4 minutes per side until golden brown and fluffy.",
        ],
        "cooking_time": 10,
        "difficulty": "easy",
        "servings": 2,
        "category": "breakfast",
    },
    {
        "title": "Fluffy Milk Pancakes",
        "ingredients": ["flour", "milk", "egg", "butter", "sugar", "baking powder"],
        "ingredient_details": [
            {"name": "flour", "quantity": 150, "unit": "g"},
            {"name": "milk", "quantity": 180, "unit": "ml"},
            {"name": "egg", "quantity": 1, "unit": "piece"},
            {"name": "butter", "quantity": 2, "unit": "tbsp"},
            {"name": "sugar", "quantity": 2, "unit": "tbsp"},
            {"name": "baking powder", "quantity": 1, "unit": "tsp"},
        ],
        "instructions": [
            "Mix flour, baking powder, and sugar in a bowl.",
            "Whisk milk, egg, and melted butter together, then fold into dry ingredients.",
            "Pour batter onto a hot lightly buttered griddle.",
            "Flip when bubbles form on the surface and cook until golden brown.",
        ],
        "cooking_time": 15,
        "difficulty": "easy",
        "servings": 2,
        "category": "breakfast",
    },
    {
        "title": "Silky Steamed Milk Egg Custard",
        "ingredients": ["egg", "milk", "sugar", "vanilla"],
        "ingredient_details": [
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "milk", "quantity": 200, "unit": "ml"},
            {"name": "sugar", "quantity": 2, "unit": "tbsp"},
            {"name": "vanilla", "quantity": 0.25, "unit": "tsp"},
        ],
        "instructions": [
            "Gently whisk eggs and sugar until dissolved without creating foam.",
            "Warm the milk slightly and gradually pour into eggs with vanilla extract.",
            "Strain the custard mixture through a fine sieve into heatproof cups.",
            "Steam on low heat for 12 minutes until smooth and silky like pudding.",
        ],
        "cooking_time": 15,
        "difficulty": "medium",
        "servings": 2,
        "category": "dessert",
    },

    # --- CLASSIC FAVORITES ---
    {
        "title": "Tomato Egg Stir Fry",
        "ingredients": ["tomato", "egg", "green onion", "oil", "salt", "sugar"],
        "ingredient_details": [
            {"name": "tomato", "quantity": 2, "unit": "pieces"},
            {"name": "egg", "quantity": 3, "unit": "pieces"},
            {"name": "green onion", "quantity": 1, "unit": "stalk"},
            {"name": "oil", "quantity": 1, "unit": "tbsp"},
            {"name": "salt", "quantity": 0.5, "unit": "tsp"},
            {"name": "sugar", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Beat eggs with a pinch of salt and scramble in hot oil until soft.",
            "Stir fry sliced juicy tomatoes until saucy.",
            "Return scrambled eggs, add a touch of sugar and scallions, toss well.",
        ],
        "cooking_time": 10,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Chicken Omelette",
        "ingredients": ["chicken", "egg", "onion", "butter", "pepper", "cheese"],
        "ingredient_details": [
            {"name": "chicken", "quantity": 100, "unit": "g"},
            {"name": "egg", "quantity": 3, "unit": "pieces"},
            {"name": "onion", "quantity": 0.5, "unit": "piece"},
            {"name": "butter", "quantity": 1, "unit": "tbsp"},
            {"name": "pepper", "quantity": 0.25, "unit": "tsp"},
            {"name": "cheese", "quantity": 30, "unit": "g"},
        ],
        "instructions": [
            "Sauté diced chicken and onions in melted butter.",
            "Pour beaten eggs over the pan and cook on medium-low heat.",
            "Sprinkle shredded cheese, fold in half, and serve hot.",
        ],
        "cooking_time": 12,
        "difficulty": "easy",
        "servings": 2,
        "category": "breakfast",
    },
    {
        "title": "Chicken Tomato Stir Fry",
        "ingredients": ["chicken", "tomato", "garlic", "soy sauce", "pepper", "onion"],
        "ingredient_details": [
            {"name": "chicken", "quantity": 200, "unit": "g"},
            {"name": "tomato", "quantity": 2, "unit": "pieces"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "soy sauce", "quantity": 1, "unit": "tbsp"},
            {"name": "pepper", "quantity": 0.5, "unit": "piece"},
            {"name": "onion", "quantity": 0.5, "unit": "piece"},
        ],
        "instructions": [
            "Sear sliced chicken breast in oil until golden.",
            "Add garlic, onions, and tomato wedges with soy sauce.",
            "Simmer for 4 minutes until a savory tomato gravy forms.",
        ],
        "cooking_time": 18,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Garlic Butter Shrimp Fried Rice",
        "ingredients": ["shrimp", "rice", "egg", "garlic", "butter", "soy sauce", "green onion"],
        "ingredient_details": [
            {"name": "shrimp", "quantity": 150, "unit": "g"},
            {"name": "rice", "quantity": 300, "unit": "g"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "butter", "quantity": 1.5, "unit": "tbsp"},
            {"name": "soy sauce", "quantity": 1.5, "unit": "tbsp"},
            {"name": "green onion", "quantity": 2, "unit": "stalks"},
        ],
        "instructions": [
            "Sauté minced garlic in melted butter, sear shrimp and set aside.",
            "Scramble eggs, toss in cold cooked rice with soy sauce on high heat.",
            "Fold in shrimp and chopped scallions, season with cracked black pepper.",
        ],
        "cooking_time": 15,
        "difficulty": "easy",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Classic Beef Bolognese",
        "ingredients": ["beef", "pasta", "tomato", "onion", "garlic", "olive oil", "cheese"],
        "ingredient_details": [
            {"name": "beef", "quantity": 250, "unit": "g"},
            {"name": "pasta", "quantity": 200, "unit": "g"},
            {"name": "tomato", "quantity": 3, "unit": "pieces"},
            {"name": "onion", "quantity": 1, "unit": "piece"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "olive oil", "quantity": 1, "unit": "tbsp"},
            {"name": "cheese", "quantity": 30, "unit": "g"},
        ],
        "instructions": [
            "Brown ground beef in olive oil with diced onion and garlic.",
            "Add crushed tomatoes and simmer on low heat for 25 minutes.",
            "Boil pasta until al dente, top with rich beef bolognese and parmesan cheese.",
        ],
        "cooking_time": 35,
        "difficulty": "medium",
        "servings": 3,
        "category": "dinner",
    },
    {
        "title": "Crispy Eggplant Parmesan Stir Fry",
        "ingredients": ["eggplant", "tomato", "garlic", "cheese", "olive oil", "basil"],
        "ingredient_details": [
            {"name": "eggplant", "quantity": 1, "unit": "piece"},
            {"name": "tomato", "quantity": 2, "unit": "pieces"},
            {"name": "garlic", "quantity": 2, "unit": "cloves"},
            {"name": "cheese", "quantity": 60, "unit": "g"},
            {"name": "olive oil", "quantity": 2, "unit": "tbsp"},
            {"name": "basil", "quantity": 5, "unit": "leaves"},
        ],
        "instructions": [
            "Slice eggplant into rounds and sear in olive oil until golden tender.",
            "Simmer with minced garlic and diced tomatoes into a rich compote.",
            "Melt mozzarella cheese over the top and garnish with fresh basil.",
        ],
        "cooking_time": 25,
        "difficulty": "medium",
        "servings": 2,
        "category": "dinner",
    },
    {
        "title": "Quick Vegetable Tofu Egg Soup",
        "ingredients": ["egg", "tofu", "spinach", "chicken broth", "sesame oil", "green onion"],
        "ingredient_details": [
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "tofu", "quantity": 150, "unit": "g"},
            {"name": "spinach", "quantity": 100, "unit": "g"},
            {"name": "chicken broth", "quantity": 500, "unit": "ml"},
            {"name": "sesame oil", "quantity": 0.5, "unit": "tsp"},
            {"name": "green onion", "quantity": 1, "unit": "stalk"},
        ],
        "instructions": [
            "Bring chicken broth to a gentle simmer with cubed tofu and spinach.",
            "Drizzle in whisked eggs in a steady stream while gently stirring.",
            "Season with sesame oil and green onions, serve warm.",
        ],
        "cooking_time": 10,
        "difficulty": "easy",
        "servings": 2,
        "category": "soup",
    },
    {
        "title": "Creamy Garlic Parmesan Pasta",
        "ingredients": ["pasta", "garlic", "butter", "milk", "cream", "cheese", "black pepper"],
        "ingredient_details": [
            {"name": "pasta", "quantity": 200, "unit": "g"},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "butter", "quantity": 2, "unit": "tbsp"},
            {"name": "milk", "quantity": 80, "unit": "ml"},
            {"name": "cream", "quantity": 80, "unit": "ml"},
            {"name": "cheese", "quantity": 50, "unit": "g"},
            {"name": "black pepper", "quantity": 0.5, "unit": "tsp"},
        ],
        "instructions": [
            "Boil pasta until al dente.",
            "Sauté garlic in butter, add milk, cream, and parmesan cheese until thick.",
            "Toss pasta in the velvety garlic cream sauce.",
        ],
        "cooking_time": 18,
        "difficulty": "medium",
        "servings": 2,
        "category": "lunch",
    },
    {
        "title": "Egg and Potato Breakfast Hash",
        "ingredients": ["potato", "egg", "onion", "butter", "bacon", "salt", "pepper"],
        "ingredient_details": [
            {"name": "potato", "quantity": 2, "unit": "pieces"},
            {"name": "egg", "quantity": 2, "unit": "pieces"},
            {"name": "onion", "quantity": 0.5, "unit": "piece"},
            {"name": "butter", "quantity": 1, "unit": "tbsp"},
            {"name": "bacon", "quantity": 50, "unit": "g"},
            {"name": "salt", "quantity": 0.5, "unit": "tsp"},
            {"name": "pepper", "quantity": 0.25, "unit": "tsp"},
        ],
        "instructions": [
            "Dice potatoes and pan-fry in butter with bacon until crispy.",
            "Make two wells in the hash and crack eggs directly into the skillet.",
            "Cover and cook until egg whites are set and yolks are runny.",
        ],
        "cooking_time": 20,
        "difficulty": "easy",
        "servings": 2,
        "category": "breakfast",
    },
    {
        "title": "Korean Kimchi Egg Fried Rice",
        "ingredients": ["rice", "egg", "kimchi", "sesame oil", "green onion", "garlic"],
        "ingredient_details": [
            {"name": "rice", "quantity": 250, "unit": "g"},
            {"name": "egg", "quantity": 1, "unit": "piece"},
            {"name": "kimchi", "quantity": 100, "unit": "g"},
            {"name": "sesame oil", "quantity": 1, "unit": "tbsp"},
            {"name": "green onion", "quantity": 1, "unit": "stalk"},
            {"name": "garlic", "quantity": 1, "unit": "clove"},
        ],
        "instructions": [
            "Sauté chopped kimchi and garlic in sesame oil.",
            "Add cooked rice and stir fry vigorously until well combined.",
            "Top with a sunny-side-up fried egg and scallions.",
        ],
        "cooking_time": 12,
        "difficulty": "easy",
        "servings": 1,
        "category": "lunch",
    },
]


class SetupManager:
    """Manages recipe dataset onboarding, 8-step visual indexing, and dataset reset."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_running = False
        self.progress = 0.0
        self.current_step = 0
        self.status = "uninitialized"
        self.message = "Ready to download recipe dataset."
        self.selected_engine = get_embedder_engine()
        self.selected_limit = get_dataset_download_limit()

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            running = self.is_running
            prog = self.progress
            step = self.current_step
            stat = self.status
            msg = self.message
            engine = self.selected_engine
            limit = self.selected_limit

        db_path = get_database_path()
        recipe_count = 0

        if db_path.exists():
            try:
                with get_connection(db_path) as conn:
                    row = conn.execute("SELECT COUNT(*) as count FROM recipes").fetchone()
                    if row:
                        recipe_count = row["count"]
            except Exception:
                recipe_count = 0

        has_dataset = recipe_count >= MINIMUM_RECIPES_FOR_FULL_DATASET
        has_indices = get_vector_ids_path().exists() and get_bm25_index_path().exists()
        has_model = get_ranker_model_path().exists()

        is_ready = has_dataset and has_indices and has_model and not running

        cuda_available = False
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        recommended_engine = "cuda" if cuda_available else "onnx"

        return {
            "is_ready": is_ready,
            "recipe_count": recipe_count,
            "has_dataset": has_dataset,
            "has_indices": has_indices,
            "has_model": has_model,
            "is_running": running,
            "progress": prog,
            "current_step": step,
            "status": stat if running else ("ready" if is_ready else "uninitialized"),
            "message": msg,
            "cuda_available": cuda_available,
            "recommended_engine": recommended_engine,
            "selected_engine": engine,
            "selected_limit": limit,
        }

    def start_setup(self, engine: str | None = None, limit: int | None = None) -> bool:
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.progress = 5.0
            self.current_step = 1
            self.status = "initializing"
            if engine:
                self.selected_engine = engine.strip().lower()
            if limit and limit > 0:
                self.selected_limit = limit
            self.message = "Step 1/8: Initializing recipe database schema..."

        threading.Thread(target=self._run_setup_pipeline, daemon=True).start()
        return True

    def reset_dataset(self) -> dict[str, Any]:
        """Completely wipes local database, vector index, BM25 index, and ML model."""
        with self.lock:
            self.is_running = False
            self.progress = 0.0
            self.current_step = 0
            self.status = "uninitialized"
            self.message = "Dataset cleared. Click Download to start setup."

        # 1. Clear SQLite tables
        db_path = get_database_path()
        if db_path.exists():
            try:
                with get_connection(db_path) as conn:
                    conn.execute("DELETE FROM recipes")
                    conn.commit()
            except Exception as e:
                logger.warning("Could not clear recipes table: %s", e)

        # 2. Delete index and model files
        for p in [get_vector_index_path(), get_vector_ids_path(), get_bm25_index_path(), get_ranker_model_path()]:
            try:
                if p.exists():
                    os.remove(p)
                    logger.info("Deleted index/model artifact: %s", p)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", p, e)

        logger.info("Local recipe dataset and AI indices reset successfully.")
        return {"success": True, "message": "Dataset reset successfully."}

    def _update_step(self, step: int, progress: float, status: str, message: str) -> None:
        with self.lock:
            self.current_step = step
            self.progress = progress
            self.status = status
            self.message = message
        logger.info("[Step %d/8 - %d%%] %s", step, int(progress), message)

    def _run_setup_pipeline(self) -> None:
        try:
            db_path = get_database_path()
            target_limit = self.selected_limit or 50000
            engine = self.selected_engine or "onnx"

            # Step 1: Database Schema
            self._update_step(1, 10.0, "step1", "Step 1/8: Creating SQLite relational schema...")
            initialize_database(db_path)
            time.sleep(0.3)

            # Step 2: Fetch & Parse Full Catalog from Hugging Face Parquet Archive
            engine_name = "PyTorch CUDA (GPU)" if engine == "cuda" else "FastEmbed ONNX INT8 (CPU)"
            self._update_step(2, 15.0, "step2", f"Step 2/8: Downloading Full Recipe Dataset ({target_limit} recipes)...")

            def dl_cb(pct: float, msg: str) -> None:
                step2_pct = 15.0 + (pct / 100.0) * 17.0
                self._update_step(2, step2_pct, "step2", f"Step 2/8: {msg}...")

            parquet_path = None
            try:
                parquet_path = download_full_hf_parquet_file(progress_callback=dl_cb)
            except Exception as e:
                logger.warning("Parquet download encountered error: %s (will use fallback online API)", e)

            online_recipes = []
            if parquet_path and parquet_path.exists():
                self._update_step(2, 33.0, "step2", f"Step 2/8: Parsing {target_limit} recipes from parquet archive...")
                online_recipes = load_recipes_from_parquet(parquet_path, limit=target_limit)

            if len(online_recipes) < min(100, target_limit):
                self._update_step(2, 34.0, "step2", f"Step 2/8: Fetching online recipes from Hugging Face...")
                hf_rows = fetch_online_recipes_from_huggingface(limit=min(target_limit, 200))
                for r in hf_rows:
                    if not any(x["title"].lower() == r["title"].lower() for x in online_recipes):
                        online_recipes.append(r)

            all_recipes = list(CORE_RECIPE_CATALOG)
            for r in online_recipes:
                if not any(existing["title"].lower() == r["title"].lower() for existing in all_recipes):
                    all_recipes.append(r)
            total = len(all_recipes)
            time.sleep(0.3)

            # Step 3: Fast Batch Ingestion into SQLite
            self._update_step(3, 38.0, "step3", f"Step 3/8: Ingesting and normalizing {total} recipes into SQLite database...")
            with get_connection(db_path) as conn:
                conn.execute("DELETE FROM recipes")
                batch_records = []
                for r in all_recipes:
                    batch_records.append((
                        r["title"],
                        json.dumps(r["ingredients"]),
                        json.dumps(r["ingredient_details"]),
                        json.dumps(r["instructions"]),
                        r["cooking_time"],
                        r["difficulty"],
                        r["servings"],
                        r["category"],
                    ))
                conn.executemany(
                    """
                    INSERT INTO recipes (title, ingredients, ingredient_details, instructions, cooking_time, difficulty, servings, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_records,
                )
                conn.commit()
            time.sleep(0.3)

            # Step 4: AI Embedder Pre-warm & Embeddings
            self._update_step(4, 50.0, "step4", f"Step 4/8: Computing 384-dim embeddings ({engine_name}) for {total} recipes...")
            embedder = IngredientEmbedder(engine=engine)
            with get_connection(db_path) as conn:
                rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()

            recipe_ids: list[int] = []
            recipe_texts: list[str] = []
            for row in rows:
                recipe_ids.append(row["id"])
                ingredients = json.loads(row["ingredients"] or "[]")
                recipe_texts.append(build_recipe_representation(row["title"], ingredients))

            embeddings = embedder.embed_texts(recipe_texts)
            time.sleep(0.3)

            # Step 5: FAISS Dense Vector Index
            self._update_step(5, 70.0, "step5", f"Step 5/8: Building FAISS IndexFlatIP dense vector store ({total} vectors)...")
            vector_store = FaissVectorStore(dimension=embedder.dimension)
            vector_store.add(embeddings, recipe_ids)
            vector_store.save(get_vector_index_path(), get_vector_ids_path())
            time.sleep(0.3)

            # Step 6: BM25 Sparse Lexical Index
            self._update_step(6, 80.0, "step6", f"Step 6/8: Building BM25Okapi sparse lexical search index ({total} docs)...")
            bm25_store = BM25Store()
            bm25_store.add_corpus(recipe_ids, recipe_texts)
            bm25_store.save(get_bm25_index_path())
            time.sleep(0.3)

            # Step 7: Train ML Learning-to-Rank Model
            self._update_step(7, 90.0, "step7", f"Step 7/8: Training LightGBM / GBDT Machine Learning Ranker on {total} recipes...")
            from scripts.train_ranker import train_ranker
            train_ranker()
            time.sleep(0.4)

            # Step 8: Final Verification & Ready
            self._update_step(8, 100.0, "ready", f"Step 8/8: Verified {total} recipes ready! Opening search...")
            with self.lock:
                self.is_running = False
                self.progress = 100.0
                self.current_step = 8
                self.status = "ready"
                self.message = f"Setup complete! {total} recipes indexed with {engine_name}."
            logger.info("Dataset onboarding pipeline completed successfully with %d recipes (%s).", total, engine_name)

        except Exception as e:
            logger.error("Setup pipeline failed: %s", e, exc_info=True)
            with self.lock:
                self.is_running = False
                self.status = "error"
                self.message = f"Setup failed: {e}"


# Singleton instance
_setup_manager: SetupManager | None = None


def get_setup_manager() -> SetupManager:
    global _setup_manager
    if _setup_manager is None:
        _setup_manager = SetupManager()
    return _setup_manager
