from __future__ import annotations

import logging
import os
import threading
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.api.assistant import router as assistant_router
from app.backend.api.ingredients import router as ingredients_router
from app.backend.api.meal_planner import router as meal_planner_router
from app.backend.api.pantry import router as pantry_router
from app.backend.api.recipes import router as recipes_router
from app.backend.config import (
    get_auto_build_index,
    get_auto_open_browser,
    get_bm25_index_path,
    get_embedding_model_name,
    get_index_build_setup,
    get_ranker_model_path,
    get_ranking_mode,
    get_retrieval_mode,
    get_vector_index_path,
    missing_finetuned_files,
    uses_finetuned_embedding,
)
from app.backend.database.init_db import initialize_database
from app.backend.database.pantry_db import ensure_pantry_tables
from app.backend.system.heartbeat import get_heartbeat_monitor
from app.backend.system.routes import router as system_router

logger = logging.getLogger("pantrysense")
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def log_search_setup() -> None:
    """Say which embedding model, index and ranker this process will use."""
    if os.getenv("PANTRYSENSE_EMBEDDING_MODEL", "").strip():
        logger.info("Embedding model %s (PANTRYSENSE_EMBEDDING_MODEL), index %s",
                    get_embedding_model_name(), get_vector_index_path())
    elif uses_finetuned_embedding():
        logger.info("Embedding model: fine-tuned e5 at %s, index %s", get_embedding_model_name(), get_vector_index_path())
    else:
        missing = ", ".join(p.name for p in missing_finetuned_files())
        logger.info("Embedding model: %s, index %s (fine-tuned e5 not used, missing: %s; see docs/embedding.md)",
                    get_embedding_model_name(), get_vector_index_path(), missing)

    if get_ranking_mode() != "ml":
        logger.info("Ranking: coverage heuristic (PANTRYSENSE_RANKING_MODE=%s)", get_ranking_mode())
    elif get_ranker_model_path().exists():
        logger.info("Ranking: LTR bundle %s", get_ranker_model_path())
    else:
        logger.info(
            "Ranking: no LTR bundle at %s, results keep the hybrid RRF order. "
            "Download it with 'python tasks.py fetch-models' or train it (docs/ltr.md).",
            get_ranker_model_path(),
        )


def _open_default_browser() -> None:
    try:
        url = "http://localhost:8000"
        logger.info("Opening default external browser at %s...", url)
        webbrowser.open_new_tab(url)
    except Exception as e:
        logger.warning("Could not auto-open browser: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    ensure_pantry_tables()

    # Build missing retrieval indices on first startup. The LTR ranker is not
    # trained here: its dataset needs ~10k hybrid searches, so a missing model
    # just means results keep the RRF order (see log_search_setup).
    if get_auto_build_index():
        mode = get_retrieval_mode()
        if mode in ("semantic", "hybrid", "lexical"):
            # The build target, not the current setup: with the fine-tuned model
            # in data/ but no e5 index yet this builds the e5 index.
            ids_path = get_index_build_setup().ids_path
            bm25_path = get_bm25_index_path()

            sem_missing = mode in ("semantic", "hybrid") and not ids_path.exists()
            bm25_missing = mode in ("lexical", "hybrid") and not bm25_path.exists()

            if sem_missing or bm25_missing:
                logger.info("Retrieval index files missing for mode '%s'. Automatically building indices...", mode)
                try:
                    from scripts.build_vector_index import build_indices
                    build_indices()
                    logger.info("Auto-built retrieval indices successfully.")
                except Exception as e:
                    logger.error("Failed to auto-build indices: %s", e)

    log_search_setup()

    # Start heartbeat monitor
    monitor = get_heartbeat_monitor()
    await monitor.start()

    # Pre-warm ingredient vocabulary in background
    from app.backend.recipes.suggester import get_ingredient_suggester
    threading.Thread(target=get_ingredient_suggester().ensure_loaded, daemon=True).start()

    # Auto-open external browser tab after startup
    if get_auto_open_browser():
        threading.Timer(0.3, _open_default_browser).start()

    yield

    # Teardown
    await monitor.stop()


app = FastAPI(title="PantrySense API", version="1.1.0", lifespan=lifespan)
app.include_router(ingredients_router)
app.include_router(recipes_router)
app.include_router(pantry_router)
app.include_router(meal_planner_router)
app.include_router(assistant_router)
app.include_router(system_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def serve_root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
