"""Dataset onboarding pipeline: ingest, precompute metadata, embed, index, train."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
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
from app.backend.database.recipe_metadata import populate_recipe_metadata
from app.backend.retrieval.bm25_store import BM25Store, bm25_index_exists
from app.backend.retrieval.embedder import IngredientEmbedder, write_index_metadata
from app.backend.retrieval.semantic import build_recipe_representation
from app.backend.retrieval.vector_store import FaissVectorStore
from app.backend.system.core_catalog import CORE_RECIPE_CATALOG
from app.backend.system.dataset_sources import (
    download_full_hf_parquet_file,
    fetch_online_recipes_from_huggingface,
    load_recipes_from_parquet,
)

logger = logging.getLogger("pantrysense.setup")

MINIMUM_RECIPES_FOR_FULL_DATASET = 15


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
        has_indices = get_vector_ids_path().exists() and bm25_index_exists(get_bm25_index_path())
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
                    try:
                        conn.execute("DELETE FROM recipe_metadata")
                    except Exception:
                        pass
                    conn.commit()
            except Exception as e:
                logger.warning("Could not clear recipes table: %s", e)

        # 2. Delete index and model files
        bm25_path = get_bm25_index_path()
        for p in [get_vector_index_path(), get_vector_ids_path(), bm25_path, bm25_path.with_suffix(".npz"), get_ranker_model_path()]:
            try:
                if p.exists():
                    os.remove(p)
                    logger.info("Deleted index/model artifact: %s", p)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", p, e)

        _drop_cached_search_components()
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
                def parse_cb(parsed_so_far: int, target: int) -> None:
                    pct = 25.0 + (parsed_so_far / target) * 10.0
                    self._update_step(2, pct, "step2", f"Step 2/8: Parsed {parsed_so_far:,} / {target:,} recipes from parquet archive...")

                self._update_step(2, 25.0, "step2", f"Step 2/8: Parsing {target_limit:,} recipes from parquet archive...")
                online_recipes = load_recipes_from_parquet(parquet_path, limit=target_limit, progress_callback=parse_cb)

            if len(online_recipes) < min(100, target_limit):
                self._update_step(2, 34.0, "step2", "Step 2/8: Fetching online recipes from Hugging Face...")
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
            self._update_step(3, 35.0, "step3", f"Step 3/8: Ingesting and normalizing {total:,} recipes into SQLite database...")
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
                    step3_pct = 35.0 + (inserted_cnt / total) * 5.0
                    self._update_step(3, step3_pct, "step3", f"Step 3/8: Ingested {inserted_cnt:,} / {total:,} recipes into SQLite...")

            # Dietary tags, allergens and nutrition are computed once here so
            # search can filter on them before retrieval.
            def meta_cb(done: int, todo: int) -> None:
                self._update_step(3, 38.0 + (done / max(todo, 1)) * 2.0, "step3",
                                  f"Step 3/8: Precomputed dietary and nutrition data for {done:,} / {todo:,} recipes...")

            populate_recipe_metadata(db_path, workers=max(1, (os.cpu_count() or 2) - 2), progress=meta_cb)
            time.sleep(0.3)

            # Step 4: AI Embedder Pre-warm & Embeddings
            self._update_step(4, 40.0, "step4", f"Step 4/8: Computing 384-dim embeddings ({engine_name}) for {total:,} recipes...")
            embedder = IngredientEmbedder(engine=engine)
            with get_connection(db_path) as conn:
                rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()

            recipe_ids: list[int] = []
            recipe_texts: list[str] = []
            for row in rows:
                recipe_ids.append(row["id"])
                ingredients = json.loads(row["ingredients"] or "[]")
                recipe_texts.append(build_recipe_representation(row["title"], ingredients))

            chunk_size = 10000
            all_embeddings = []
            for idx in range(0, total, chunk_size):
                sub_texts = recipe_texts[idx : idx + chunk_size]
                sub_emb = embedder.embed_texts(sub_texts)
                all_embeddings.append(sub_emb)
                done_count = min(total, idx + len(sub_texts))
                pct = 40.0 + (done_count / total) * 28.0
                self._update_step(4, pct, "step4", f"Step 4/8: Computed {done_count:,} / {total:,} embeddings ({engine_name})...")

            import numpy as np
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, embedder.dimension), dtype=np.float32)
            time.sleep(0.3)

            # Step 5: FAISS Dense Vector Index
            self._update_step(5, 70.0, "step5", f"Step 5/8: Building FAISS IndexFlatIP dense vector store ({total:,} vectors)...")
            vector_store = FaissVectorStore(dimension=embedder.dimension)
            vector_store.add(embeddings, recipe_ids)
            vector_store.save(get_vector_index_path(), get_vector_ids_path())
            write_index_metadata(get_vector_index_path(), embedder, count=len(recipe_ids))
            time.sleep(0.3)

            # Step 6: BM25 Sparse Lexical Index
            self._update_step(6, 80.0, "step6", f"Step 6/8: Building BM25Okapi sparse lexical search index ({total:,} docs)...")
            bm25_store = BM25Store()
            bm25_store.add_corpus(recipe_ids, recipe_texts)
            bm25_path = get_bm25_index_path()
            bm25_store.save(bm25_path.with_suffix(".npz"))
            if bm25_path.suffix.lower() != ".npz":
                # Kept for code that still checks for the JSON file.
                bm25_store.save(bm25_path)
            time.sleep(0.3)

            # Step 7: Train ML Learning-to-Rank Model
            self._update_step(7, 90.0, "step7", f"Step 7/8: Training LightGBM / GBDT Machine Learning Ranker on {total} recipes...")
            from scripts.train_ranker import train_ranker
            train_ranker()
            time.sleep(0.4)

            # Step 8: Final Verification & Ready
            self._update_step(8, 100.0, "ready", f"Step 8/8: Verified {total} recipes ready! Opening search...")

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
