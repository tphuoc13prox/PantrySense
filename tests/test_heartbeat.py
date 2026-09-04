from __future__ import annotations

import asyncio
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from app.backend.database.init_db import initialize_database
from app.backend.main import app
from app.backend.system.heartbeat import HeartbeatMonitor


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "recipes.db"
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(database_path))
    monkeypatch.setenv("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    monkeypatch.setenv("PANTRYSENSE_AUTO_SHUTDOWN", "false")
    monkeypatch.setenv("PANTRYSENSE_RETRIEVAL_MODE", "rule_based")
    initialize_database(database_path)
    return TestClient(app)


def test_heartbeat_ping_and_leave(client: TestClient) -> None:
    # Send heartbeat from tab_1
    res1 = client.post("/api/system/heartbeat", json={"tab_id": "tab_1"})
    assert res1.status_code == 200
    assert res1.json()["status"] == "ok"
    assert res1.json()["active_tabs"] >= 1

    # Send heartbeat from tab_2
    res2 = client.post("/api/system/heartbeat", json={"tab_id": "tab_2"})
    assert res2.status_code == 200
    assert res2.json()["active_tabs"] >= 2

    # Leave tab_1
    res3 = client.post("/api/system/heartbeat/leave", json={"tab_id": "tab_1"})
    assert res3.status_code == 200

    # Leave tab_2
    res4 = client.post("/api/system/heartbeat/leave", json={"tab_id": "tab_2"})
    assert res4.status_code == 200


def test_heartbeat_monitor_tab_ttl() -> None:
    monitor = HeartbeatMonitor(tab_ttl=0.1)
    monitor.record_heartbeat("tab_x")
    assert monitor.get_active_tab_count() == 1

    # Wait for TTL to expire
    time.sleep(0.15)
    assert monitor.get_active_tab_count() == 0


def test_heartbeat_monitor_triggers_shutdown_after_inactivity() -> None:
    async def _async_run():
        shutdown_triggered = False

        def on_shutdown() -> None:
            nonlocal shutdown_triggered
            shutdown_triggered = True

        monitor = HeartbeatMonitor(
            tab_ttl=0.1,
            inactive_timeout=0.2,
            grace_period=0.1,
            on_shutdown=on_shutdown,
        )

        await monitor.start()

        # Record heartbeat
        monitor.record_heartbeat("test_tab")
        assert monitor.get_active_tab_count() == 1

        # Unregister tab
        monitor.unregister_tab("test_tab")
        assert monitor.get_active_tab_count() == 0

        # Wait for inactive timeout (0.2s + margin)
        await asyncio.sleep(0.4)

        assert shutdown_triggered is True
        await monitor.stop()

    asyncio.run(_async_run())
