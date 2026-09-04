from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.backend.system.dataset_loader import get_setup_manager
from app.backend.system.heartbeat import get_heartbeat_monitor

router = APIRouter(prefix="/api/system", tags=["system"])


class HeartbeatRequest(BaseModel):
    tab_id: str = Field(..., description="Unique client browser tab identifier")


class HeartbeatResponse(BaseModel):
    status: str = "ok"
    active_tabs: int


class SetupStatusResponse(BaseModel):
    is_ready: bool
    recipe_count: int
    has_dataset: bool
    has_indices: bool
    has_model: bool
    is_running: bool
    progress: float
    current_step: int = 0
    status: str
    message: str
    cuda_available: bool = False
    recommended_engine: str = "onnx"
    selected_engine: str = "onnx"
    selected_limit: int = 50000


class StartSetupRequest(BaseModel):
    engine: str | None = Field(None, description="Embedding acceleration engine: 'cuda' or 'onnx'")
    dataset_limit: int | None = Field(None, description="Number of recipes to download/ingest")


class SetEngineRequest(BaseModel):
    engine: str = Field(..., description="Embedding acceleration engine: 'cuda' or 'onnx'")


class GenericActionResponse(BaseModel):
    success: bool
    message: str


@router.post("/heartbeat", response_model=HeartbeatResponse)
def ping_heartbeat(request: HeartbeatRequest) -> HeartbeatResponse:
    monitor = get_heartbeat_monitor()
    count = monitor.record_heartbeat(request.tab_id)
    return HeartbeatResponse(status="ok", active_tabs=count)


@router.post("/heartbeat/leave", response_model=HeartbeatResponse)
def leave_heartbeat(request: HeartbeatRequest) -> HeartbeatResponse:
    monitor = get_heartbeat_monitor()
    count = monitor.unregister_tab(request.tab_id)
    return HeartbeatResponse(status="ok", active_tabs=count)


@router.get("/setup-status", response_model=SetupStatusResponse)
def get_setup_status() -> SetupStatusResponse:
    manager = get_setup_manager()
    info = manager.get_status()
    return SetupStatusResponse(**info)


@router.post("/setup-init", response_model=GenericActionResponse)
def trigger_setup_init(request: StartSetupRequest | None = None) -> GenericActionResponse:
    manager = get_setup_manager()
    engine = request.engine if request else None
    limit = request.dataset_limit if request else None
    started = manager.start_setup(engine=engine, limit=limit)
    if started:
        return GenericActionResponse(success=True, message="Setup process initialized.")
    return GenericActionResponse(success=False, message="Setup is already in progress.")


@router.post("/set-engine", response_model=GenericActionResponse)
def set_engine_endpoint(request: SetEngineRequest) -> GenericActionResponse:
    from app.backend.config import set_embedder_engine
    set_embedder_engine(request.engine)
    manager = get_setup_manager()
    with manager.lock:
        manager.selected_engine = request.engine.strip().lower()
    return GenericActionResponse(success=True, message=f"Engine set to {request.engine}")


@router.post("/reset-dataset", response_model=GenericActionResponse)
def reset_dataset_endpoint() -> GenericActionResponse:
    manager = get_setup_manager()
    result = manager.reset_dataset()
    return GenericActionResponse(**result)
