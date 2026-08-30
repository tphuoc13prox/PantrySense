from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import get_database_path, get_vector_ids_path, get_vector_index_path
from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.retrieval.embedder import IngredientEmbedder
from app.backend.retrieval.semantic import build_recipe_representation
from app.backend.retrieval.vector_store import FaissVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_vector_index(
    db_path: Path | None = None,
    index_path: Path | None = None,
    ids_path: Path | None = None,
) -> int:
    target_db = db_path or get_database_path()
    target_index = index_path or get_vector_index_path()
    target_ids = ids_path or get_vector_ids_path()

    logger.info("Initializing database at %s (if not present)...", target_db)
    initialize_database(target_db)

    logger.info("Loading recipes from database...")
    with get_connection(target_db) as conn:
        rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()

    if not rows:
        logger.warning("No recipes found in the database. Vector index will be empty.")
        return 0

    recipe_ids: list[int] = []
    recipe_texts: list[str] = []

    for row in rows:
        r_id = row["id"]
        title = row["title"]
        ingredients = json.loads(row["ingredients"] or "[]")
        text_repr = build_recipe_representation(title, ingredients)

        recipe_ids.append(r_id)
        recipe_texts.append(text_repr)

    logger.info("Loaded %d recipes. Initializing IngredientEmbedder...", len(recipe_ids))
    start_time = time.perf_counter()
    embedder = IngredientEmbedder()

    logger.info(
        "Generating embeddings with model '%s' (dimension: %d)...",
        embedder.model_name,
        embedder.dimension,
    )
    embeddings = embedder.embed_texts(recipe_texts)
    embedding_duration = time.perf_counter() - start_time
    logger.info("Generated %d embeddings in %.2f seconds.", len(embeddings), embedding_duration)

    logger.info("Building vector index...")
    vector_store = FaissVectorStore(dimension=embedder.dimension)
    vector_store.add(embeddings, recipe_ids)

    logger.info("Saving index to %s and IDs to %s...", target_index, target_ids)
    vector_store.save(target_index, target_ids)

    logger.info("=" * 60)
    logger.info("Vector Index Build Summary:")
    logger.info("  Total Indexed Recipes : %d", len(recipe_ids))
    logger.info("  Embedding Dimension   : %d", embedder.dimension)
    logger.info("  Vector Index Path     : %s", target_index)
    logger.info("  Vector IDs Path       : %s", target_ids)
    logger.info("=" * 60)

    return len(recipe_ids)


if __name__ == "__main__":
    build_vector_index()
