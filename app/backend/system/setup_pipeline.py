"""Dataset onboarding pipeline: ingest, precompute metadata, embed, index."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.backend.config import (
    get_bm25_index_path,
    get_database_path,
    get_dataset_download_limit,
    get_embedder_engine,
    get_index_build_setup,
    get_ranker_model_path,
    get_vector_ids_path,
    get_vector_index_path,
    has_finetuned_model,
)
from app.backend.database.connection import get_connection
from app.backend.database.init_db import initialize_database
from app.backend.database.recipe_metadata import populate_recipe_metadata
from app.backend.retrieval.bm25_store import bm25_index_exists
from app.backend.retrieval.embedder import index_metadata_path
from app.backend.retrieval.index_build import build_search_indices
from app.backend.system.core_catalog import CORE_RECIPE_CATALOG
from app.backend.system.dataset_sources import (
    download_full_hf_parquet_file,
    fetch_online_recipes_from_huggingface,
    import_recipe_sources,
    load_recipes_from_parquet,
)

logger = logging.getLogger("pantrysense.setup")

MINIMUM_RECIPES_FOR_FULL_DATASET = 15

# (step, progress percent, message)
StepReport = Callable[[int, float, str], None]


def _log_step(step: int, progress: float, message: str) -> None:
    logger.info("[Step %d/8 - %d%%] %s", step, int(progress), message)


def indices_ready() -> bool:
    """True when BM25 and the dense index the app should use are built.

    With the fine-tuned e5 model present that means the e5 index, so a MiniLM
    index left from before the model was added does not count.
    """
    setup = get_index_build_setup()
    dense = setup.index_path.exists() and setup.ids_path.exists() and index_metadata_path(setup.index_path).exists()
    return dense and bm25_index_exists(get_bm25_index_path())


def describe_embedding(engine: str) -> str:
    """Short label for the embedding model and backend a build will use."""
    if get_index_build_setup().name == "e5-finetuned":
        return "fine-tuned e5-small, " + ("GPU" if engine == "cuda" else "CPU")
    return "MiniLM, " + ("PyTorch CUDA (GPU)" if engine == "cuda" else "FastEmbed ONNX (CPU)")


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
        has_indices = indices_ready()
        has_model = get_ranker_model_path().exists()

        # The LTR model is optional: without it results keep the RRF order.
        is_ready = has_dataset and has_indices and not running

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
            "has_finetuned_model": has_finetuned_model(),
            "embedding_setup": get_index_build_setup().name,
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

    def _claim(self, engine: str | None, limit: int | None) -> bool:
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
        return True

    def start_setup(self, engine: str | None = None, limit: int | None = None) -> bool:
        if not self._claim(engine, limit):
            return False
        threading.Thread(target=self._run_setup_pipeline, daemon=True).start()
        return True

    def run_blocking(self, engine: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """Run the pipeline in the calling thread (scripts, smoke test); returns get_status()."""
        if not self._claim(engine, limit):
            raise RuntimeError("setup is already running")
        self._run_setup_pipeline()
        return self.get_status()

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
                    try:
                        conn.execute("DELETE FROM recipe_metadata")
                    except Exception:
                        pass
                    conn.commit()
            except Exception as e:
                logger.warning("Could not clear recipes table: %s", e)

        # 2. Delete index and model files. The fine-tuned model folder itself is
        # kept: it is downloaded, not built here.
        bm25_path = get_bm25_index_path()
        build = get_index_build_setup()
        vector_files = {get_vector_index_path(), get_vector_ids_path(), build.index_path, build.ids_path}
        vector_files |= {index_metadata_path(p) for p in (get_vector_index_path(), build.index_path)}
        for p in [*sorted(vector_files), bm25_path, bm25_path.with_suffix(".npz"), get_ranker_model_path()]:
            try:
                if p.exists():
                    os.remove(p)
                    logger.info("Deleted index/model artifact: %s", p)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", p, e)

        _drop_cached_search_components()
        logger.info("Local recipe dataset and AI indices reset successfully.")
        return {"success": True, "message": "Dataset reset successfully."}

    def _update_step(self, step: int, progress: float, message: str) -> None:
        status = "ready" if step == 8 else f"step{step}"
        with self.lock:
            self.current_step = step
            self.progress = progress
            self.status = status
            self.message = message
        _log_step(step, progress, message)

    def _run_setup_pipeline(self) -> None:
        try:
            db_path = get_database_path()
            engine = self.selected_engine or "onnx"
            total = ingest_dataset(db_path, self.selected_limit or 50000, self._update_step)
            engine_name = build_indices_step(db_path, engine, total, self._update_step)

            # Step 7: the LTR ranker is not trained here (its dataset needs ~10k
            # hybrid searches); without a bundle, results keep the RRF order.
            if get_ranker_model_path().exists():
                step7 = "Step 7/8: Using the existing learning-to-rank model."
            else:
                step7 = ("Step 7/8: No learning-to-rank model; results use hybrid RRF order "
                         "(python tasks.py fetch-models downloads it).")
            self._update_step(7, 90.0, step7)
            time.sleep(0.4)

            # Step 8: Final Verification & Ready
            self._update_step(8, 100.0, f"Step 8/8: Verified {total} recipes ready! Opening search...")

            _drop_cached_search_components()

            # Reload ingredient suggester vocabulary from the newly populated database
            try:
                from app.backend.recipes.suggester import get_ingredient_suggester
                get_ingredient_suggester(db_path=db_path, force_reload=True)
            except Exception as e:
                logger.warning("Could not reload ingredient suggester after setup: %s", e)

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


def ingest_dataset(db_path: Path, target_limit: int, report: StepReport = _log_step) -> int:
    """Steps 1-3: schema, dataset download and parse, SQLite ingest, precomputed metadata.

    Returns the number of recipes stored.
    """
    # Step 1: Database Schema
    report(1, 10.0, "Step 1/8: Creating SQLite relational schema...")
    initialize_database(db_path)

    # Step 2: Fetch & Parse Full Catalog from Hugging Face Parquet Archive
    report(2, 15.0, f"Step 2/8: Downloading Full Recipe Dataset ({target_limit} recipes)...")

    def dl_cb(pct: float, msg: str) -> None:
        report(2, 15.0 + (pct / 100.0) * 17.0, f"Step 2/8: {msg}...")

    parquet_path = None
    try:
        parquet_path = download_full_hf_parquet_file(progress_callback=dl_cb)
    except Exception as e:
        logger.warning("Parquet download encountered error: %s (will use fallback online API)", e)

    online_recipes = []
    if parquet_path and parquet_path.exists():
        def parse_cb(parsed_so_far: int, target: int) -> None:
            report(2, 25.0 + (parsed_so_far / target) * 10.0,
                   f"Step 2/8: Parsed {parsed_so_far:,} / {target:,} recipes from parquet archive...")

        report(2, 25.0, f"Step 2/8: Parsing {target_limit:,} recipes from parquet archive...")
        online_recipes = load_recipes_from_parquet(parquet_path, limit=target_limit, progress_callback=parse_cb)

    if len(online_recipes) < min(100, target_limit):
        report(2, 34.0, "Step 2/8: Fetching online recipes from Hugging Face...")
        hf_rows = fetch_online_recipes_from_huggingface(limit=min(target_limit, 200))
        for r in hf_rows:
            if not any(x["title"].lower() == r["title"].lower() for x in online_recipes):
                online_recipes.append(r)

    all_recipes = list(CORE_RECIPE_CATALOG)
    seen_titles = {r["title"].strip().lower() for r in all_recipes}
    for r in online_recipes:
        clean_t = r["title"].strip().lower()
        if clean_t not in seen_titles:
            seen_titles.add(clean_t)
            all_recipes.append(r)
    total = len(all_recipes)

    # Step 3: Fast Batch Ingestion into SQLite
    report(3, 35.0, f"Step 3/8: Ingesting and normalizing {total:,} recipes into SQLite database...")
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
        chunk_size = 25000
        for c_idx in range(0, len(batch_records), chunk_size):
            slice_records = batch_records[c_idx : c_idx + chunk_size]
            conn.executemany(
                """
                INSERT INTO recipes (title, ingredients, ingredient_details, instructions, cooking_time, difficulty, servings, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                slice_records,
            )
            conn.commit()
            inserted_cnt = min(total, c_idx + len(slice_records))
            report(3, 35.0 + (inserted_cnt / total) * 3.0,
                   f"Step 3/8: Ingested {inserted_cnt:,} / {total:,} recipes into SQLite...")

    # Dietary tags, allergens and nutrition are computed once here so
    # search can filter on them before retrieval.
    def meta_cb(done: int, todo: int) -> None:
        report(3, 38.0 + (done / max(todo, 1)) * 2.0,
               f"Step 3/8: Precomputed dietary and nutrition data for {done:,} / {todo:,} recipes...")

    if parquet_path and parquet_path.exists():
        # Keywords, category and the ingredients past the stored twelve feed the dietary tags.
        try:
            import_recipe_sources(db_path, parquet_path)
        except Exception as e:
            logger.warning("Could not read recipe keywords from the parquet: %s", e)
    populate_recipe_metadata(db_path, workers=max(1, (os.cpu_count() or 2) - 2), progress=meta_cb)
    return total


def build_indices_step(db_path: Path, engine: str, total: int, report: StepReport = _log_step) -> str:
    """Steps 4-6: embeddings, FAISS and BM25 for the setup get_index_build_setup() picks.

    The vector index is written where resolve_embedding_setup() looks for it, so
    with the fine-tuned e5 model present the app moves to e5 as soon as this
    finishes. Returns a short label for the embedding setup.
    """
    engine_name = describe_embedding(engine)
    report(4, 40.0, f"Step 4/8: Computing embeddings ({engine_name}) for {total:,} recipes...")

    def progress(stage: str, done: int, todo: int) -> None:
        if stage == "embed":
            report(4, 40.0 + (done / max(todo, 1)) * 28.0,
                   f"Step 4/8: Computed {done:,} / {todo:,} embeddings ({engine_name})...")
        elif stage == "faiss":
            report(5, 70.0, f"Step 5/8: Building FAISS dense vector index ({todo:,} vectors)...")
        else:
            report(6, 80.0, f"Step 6/8: Building BM25 sparse lexical index ({todo:,} docs)...")

    build_search_indices(db_path, get_index_build_setup(), engine=engine, progress=progress)
    return engine_name


def _drop_cached_search_components() -> None:
    try:
        from app.backend.recipes.service import reset_search_components

        reset_search_components()
    except Exception as e:
        logger.warning("Could not reset cached search components: %s", e)


# Singleton instance
_setup_manager: SetupManager | None = None


def get_setup_manager() -> SetupManager:
    global _setup_manager
    if _setup_manager is None:
        _setup_manager = SetupManager()
    return _setup_manager
