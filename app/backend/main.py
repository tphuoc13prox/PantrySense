from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
from typing import AsyncIterator
import webbrowser

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.api.recipes import router as recipes_router
from app.backend.config import (
    get_auto_open_browser,
    get_bm25_index_path,
    get_retrieval_mode,
    get_vector_ids_path,
    get_vector_index_path,
)
from app.backend.database.init_db import initialize_database
from app.backend.system.heartbeat import get_heartbeat_monitor
from app.backend.system.routes import router as system_router

logger = logging.getLogger("pantrysense")
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


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

    # Smart Auto-Build on first startup if indices are missing
    mode = get_retrieval_mode()
    if mode in ("semantic", "hybrid", "lexical"):
        ids_path = get_vector_ids_path()
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
        else:
            logger.info("Retrieval indices verified for mode '%s'.", mode)

    # Start heartbeat monitor
    monitor = get_heartbeat_monitor()
    await monitor.start()

    # Auto-open external browser tab after startup
    if get_auto_open_browser():
        threading.Timer(0.3, _open_default_browser).start()

    yield

    # Teardown
    await monitor.stop()


app = FastAPI(title="PantrySense API", version="0.5.0", lifespan=lifespan)
app.include_router(recipes_router)
app.include_router(system_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def serve_root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
