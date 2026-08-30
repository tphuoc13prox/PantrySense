from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from typing import Callable

from app.backend.config import (
    get_auto_shutdown_enabled,
    get_inactive_timeout_seconds,
    get_startup_grace_period_seconds,
)

logger = logging.getLogger("pantrysense.heartbeat")


class HeartbeatMonitor:
    """Monitors active browser tabs and initiates server shutdown after inactivity."""

    def __init__(
        self,
        tab_ttl: float = 6.0,
        inactive_timeout: float | None = None,
        grace_period: float | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self.tab_ttl = tab_ttl
        self.inactive_timeout = (
            inactive_timeout if inactive_timeout is not None else get_inactive_timeout_seconds()
        )
        self.grace_period = (
            grace_period if grace_period is not None else get_startup_grace_period_seconds()
        )
        self._on_shutdown = on_shutdown or self._default_shutdown

        self._tabs: dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._last_active_time = time.time()
        self._has_ever_had_tabs = False
        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None

    def record_heartbeat(self, tab_id: str) -> int:
        """Record or refresh a heartbeat from a browser tab."""
        now = time.time()
        with self._lock:
            self._tabs[tab_id] = now
            self._last_active_time = now
            self._has_ever_had_tabs = True
            return self._prune_and_count_locked(now)

    def unregister_tab(self, tab_id: str) -> int:
        """Explicitly unregister a tab that is closing."""
        now = time.time()
        with self._lock:
            self._tabs.pop(tab_id, None)
            count = self._prune_and_count_locked(now)
            if count == 0:
                self._last_active_time = now
            return count

    def get_active_tab_count(self) -> int:
        """Return the number of currently active tabs."""
        now = time.time()
        with self._lock:
            return self._prune_and_count_locked(now)

    def _prune_and_count_locked(self, now: float) -> int:
        """Remove expired tabs and return the active count."""
        expired = [tid for tid, last_seen in self._tabs.items() if now - last_seen > self.tab_ttl]
        for tid in expired:
            del self._tabs[tid]
        return len(self._tabs)

    async def start(self) -> None:
        """Start the background monitor loop."""
        if not get_auto_shutdown_enabled():
            logger.info("Auto-shutdown is disabled by configuration.")
            return

        self._running = True
        self._start_time = time.time()
        self._last_active_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Heartbeat monitor started (grace_period=%.1fs, inactive_timeout=%.1fs).",
            self.grace_period,
            self.inactive_timeout,
        )

    async def stop(self) -> None:
        """Stop the background monitor loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Periodic loop checking tab activity."""
        while self._running:
            await asyncio.sleep(1.0)
            now = time.time()
            uptime = now - self._start_time
            active_count = self.get_active_tab_count()

            if active_count > 0:
                with self._lock:
                    self._last_active_time = now
                continue

            # No active tabs
            if not self._has_ever_had_tabs:
                # Still in startup grace period?
                if uptime < self.grace_period:
                    continue
                inactive_duration = uptime - self.grace_period
            else:
                with self._lock:
                    inactive_duration = now - self._last_active_time

            if inactive_duration >= self.inactive_timeout:
                logger.warning(
                    "No active browser tabs detected for %.1f seconds. Initiating automatic server shutdown.",
                    inactive_duration,
                )
                self._running = False
                self._on_shutdown()
                break

    @staticmethod
    def _default_shutdown() -> None:
        """Terminate the server process gracefully."""
        pid = os.getpid()
        logger.info("Sending termination signal to PID %d...", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGINT)
            except Exception as e:
                logger.error("Failed to terminate process %d: %e", pid, e)


# Global singleton instance
_global_monitor: HeartbeatMonitor | None = None


def get_heartbeat_monitor() -> HeartbeatMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HeartbeatMonitor()
    return _global_monitor
