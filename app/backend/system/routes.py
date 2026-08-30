from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.backend.system.heartbeat import get_heartbeat_monitor

router = APIRouter(prefix="/api/system", tags=["system"])


class HeartbeatRequest(BaseModel):
    tab_id: str = Field(..., description="Unique client browser tab identifier")


class HeartbeatResponse(BaseModel):
    status: str = "ok"
    active_tabs: int


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
