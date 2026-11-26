"""Building the FAISS and BM25 indices from recipes.db.

Shared by the setup pipeline, app startup and scripts/build_vector_index.py, so
all three pick the embedding model the same way (config.get_index_build_setup)
and write the index where the app will look for it.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.backend.config import EmbeddingSetup, get_bm25_index_path, get_database_path, get_index_build_setup
from app.backend.database.connection import get_connection
from app.backend.retrieval.bm25_store import BM25Store
from app.backend.retrieval.embedder import IngredientEmbedder, write_index_metadata
from app.backend.retrieval.semantic import build_recipe_representation
from app.backend.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

# (stage, done, total): stage is "embed", "faiss" or "bm25".
Progress = Callable[[str, int, int], None]

EMBED_CHUNK = 10_000


@dataclass(frozen=True)
class IndexBuild:
    setup: EmbeddingSetup
    recipes: int
    dimension: int
    seconds: float


def load_recipe_texts(db_path: Path) -> tuple[list[int], list[str]]:
    """Recipe ids and the text that is embedded and BM25-indexed for each."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()
    ids = [int(row["id"]) for row in rows]
    texts = [build_recipe_representation(row["title"], json.loads(row["ingredients"] or "[]")) for row in rows]
    return ids, texts


def build_search_indices(
    db_path: Path | None = None,
    setup: EmbeddingSetup | None = None,
    bm25_path: Path | None = None,
    engine: str | None = None,
    legacy_json: bool = True,
    progress: Progress | None = None,
) -> IndexBuild:
    """Embed every recipe, then write the FAISS index, its metadata and BM25.

    BM25 is written as ``.npz`` next to ``bm25_path``. When that path is a
    ``.json`` file the JSON form is written too (``legacy_json``), since
    existence checks elsewhere still look for it.
    """
    start = time.perf_counter()
    db_path = db_path or get_database_path()
    setup = setup or get_index_build_setup()
    bm25_path = bm25_path or get_bm25_index_path()

    ids, texts = load_recipe_texts(db_path)
    if not ids:
        logger.warning("No recipes in %s; no index written.", db_path)
        return IndexBuild(setup, 0, 0, 0.0)
    embedder = IngredientEmbedder(model_name=setup.model, engine=engine)
    logger.info("Embedding %d recipes with %s (%s)", len(ids), setup.model, setup.name)

    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), EMBED_CHUNK):
        chunks.append(embedder.embed_texts(texts[i : i + EMBED_CHUNK]))
        if progress:
            progress("embed", min(len(texts), i + EMBED_CHUNK), len(texts))
    embeddings = np.vstack(chunks) if chunks else np.empty((0, embedder.dimension), dtype=np.float32)

    if progress:
        progress("faiss", 0, len(ids))
    store = FaissVectorStore(dimension=embedder.dimension)
    store.add(embeddings, ids)
    store.save(setup.index_path, setup.ids_path)
    write_index_metadata(setup.index_path, embedder, count=len(ids))

    if progress:
        progress("bm25", 0, len(ids))
    bm25 = BM25Store()
    bm25.add_corpus(ids, texts)
    bm25.save(bm25_path.with_suffix(".npz"))
    if bm25_path.suffix.lower() != ".npz" and legacy_json:
        bm25.save(bm25_path)

    seconds = time.perf_counter() - start
    logger.info("Indexed %d recipes (%s, dim %d) in %.1fs: %s, %s, %s", len(ids), setup.name,
                embedder.dimension, seconds, setup.index_path, setup.ids_path, bm25_path)
    return IndexBuild(setup, len(ids), embedder.dimension, seconds)
