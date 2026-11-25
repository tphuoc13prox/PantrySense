"""Compatibility module; the setup code now lives in smaller modules.

- recipe_parsing: raw dataset row -> recipe dict
- dataset_sources: parquet download/reader and the HF rows API
- core_catalog: built-in recipes
- setup_pipeline: SetupManager and get_setup_manager
"""
from __future__ import annotations

from app.backend.system.core_catalog import CORE_RECIPE_CATALOG
from app.backend.system.dataset_sources import (
    download_full_hf_parquet_file,
    fetch_online_recipes_from_huggingface,
    load_recipes_from_parquet,
)
from app.backend.system.recipe_parsing import (
    _parse_c_vector,
    _parse_duration_minutes,
    _parse_hf_recipe_text,
    _parse_quantity,
)
from app.backend.system.setup_pipeline import (
    MINIMUM_RECIPES_FOR_FULL_DATASET,
    SetupManager,
    get_setup_manager,
)

__all__ = [
    "CORE_RECIPE_CATALOG",
    "MINIMUM_RECIPES_FOR_FULL_DATASET",
    "SetupManager",
    "_parse_c_vector",
    "_parse_duration_minutes",
    "_parse_hf_recipe_text",
    "_parse_quantity",
    "download_full_hf_parquet_file",
    "fetch_online_recipes_from_huggingface",
    "get_setup_manager",
    "load_recipes_from_parquet",
]
