"""Fetching the recipe dataset: cached parquet download, parquet reader, HF rows API."""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from app.backend.config import get_database_path
from app.backend.system.recipe_parsing import (
    _parse_c_vector,
    _parse_duration_minutes,
    _parse_hf_recipe_text,
    _parse_quantity,
)

logger = logging.getLogger("pantrysense.setup")


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

    for batch in pf.iter_batches(batch_size=10000, columns=columns):
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
                            q_val = _parse_quantity(raw_q[idx])
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

            if limit > 0 and len(recipes) >= limit:
                break

        if progress_callback:
            progress_callback(len(recipes), limit)

        if limit > 0 and len(recipes) >= limit:
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
