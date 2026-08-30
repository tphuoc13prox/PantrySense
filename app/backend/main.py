from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.api.recipes import router as recipes_router
from app.backend.config import get_retrieval_mode, get_vector_ids_path, get_vector_index_path
from app.backend.database.init_db import initialize_database

logger = logging.getLogger("pantrysense")
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


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

    yield


app = FastAPI(title="PantrySense API", version="0.4.0", lifespan=lifespan)
app.include_router(recipes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def serve_root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
