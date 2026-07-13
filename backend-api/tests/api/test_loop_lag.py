from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from structlog.testing import capture_logs

from api.lifespan import loop_lag_probe


async def _run_probe_while(action_seconds: float, block_seconds: float = 0.0) -> list[dict]:
    with capture_logs() as logs:
        task = asyncio.create_task(loop_lag_probe(interval_s=0.05))
        await asyncio.sleep(0.1)
        if block_seconds:
            time.sleep(block_seconds)
        await asyncio.sleep(action_seconds)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    return [e for e in logs if e["event"] == "event_loop_lag"]


async def test_loop_lag_probe_fires_when_loop_is_blocked() -> None:
    events = await _run_probe_while(action_seconds=0.1, block_seconds=0.2)

    assert events
    assert events[0]["lag_ms"] > 50


async def test_loop_lag_probe_is_silent_on_an_idle_loop() -> None:
    events = await _run_probe_while(action_seconds=0.3)

    assert events == []
