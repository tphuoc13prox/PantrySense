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

    # Check semantic retrieval index status on startup
    if get_retrieval_mode() == "semantic":
        index_path = get_vector_index_path()
        ids_path = get_vector_ids_path()
        if not ids_path.exists():
            logger.warning(
                "Semantic index not found at '%s'. Run 'python scripts/build_vector_index.py' to build the index.",
                index_path,
            )
        else:
            logger.info("Semantic vector index detected at '%s'.", index_path)

    # Start heartbeat monitor
    monitor = get_heartbeat_monitor()
    await monitor.start()

    # Auto-open external browser tab after startup
    if get_auto_open_browser():
        threading.Timer(0.3, _open_default_browser).start()

    yield

    # Teardown
    await monitor.stop()


app = FastAPI(title="PantrySense API", version="0.4.1", lifespan=lifespan)
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
