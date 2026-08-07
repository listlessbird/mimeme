from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from mimeme import search
from mimeme.api import lifespan
from mimeme.api.lifespan import search_reconcile_probe


class _Search:
    def __init__(self, *, ready: bool, error: bool = False) -> None:
        self._ready = ready
        self._error = error
        self.calls = 0

    async def status(self) -> search.Status:
        self.calls += 1
        if self._error:
            raise search.Unavailable("compute is down")
        return search.Status(ready=self._ready, serving_version="v1" if self._ready else None)


async def _run(monkeypatch, client: _Search, reconcile) -> list[dict]:
    monkeypatch.setattr(lifespan.index, "reconcile", reconcile)
    env = SimpleNamespace(search=client, db=object(), artifacts=object())
    with capture_logs() as logs:
        task = asyncio.create_task(search_reconcile_probe(env, 0.01))
        await asyncio.sleep(0.08)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    return logs


async def test_probe_reconciles_when_compute_has_dropped_its_index(monkeypatch) -> None:
    seen = []

    async def reconcile(db, artifacts, remote):
        seen.append(remote)
        return search.Status(ready=True, serving_version="v1")

    client = _Search(ready=False)
    logs = await _run(monkeypatch, client, reconcile)

    assert seen
    assert seen[0] is client
    assert [e for e in logs if e["event"] == "search_reconciled"]


async def test_probe_stays_idle_while_compute_is_serving(monkeypatch) -> None:
    async def reconcile(db, artifacts, remote):
        raise AssertionError("a serving compute must not be reconciled")

    client = _Search(ready=True)
    logs = await _run(monkeypatch, client, reconcile)

    assert client.calls > 1
    assert logs == []


async def test_probe_skips_quietly_while_compute_is_unreachable(monkeypatch) -> None:
    async def reconcile(db, artifacts, remote):
        raise AssertionError("an unreachable compute must not be reconciled")

    logs = await _run(monkeypatch, _Search(ready=False, error=True), reconcile)

    assert logs == []


async def test_probe_survives_a_failing_reconcile(monkeypatch) -> None:
    calls = 0

    async def reconcile(db, artifacts, remote):
        nonlocal calls
        calls += 1
        raise ValueError("artifact checksum mismatch")

    logs = await _run(monkeypatch, _Search(ready=False), reconcile)

    assert calls > 1
    assert [e for e in logs if e["event"] == "search_reconcile_failed"]


@pytest.mark.parametrize("ready", [True, False])
async def test_probe_never_reconciles_before_its_first_interval(monkeypatch, ready) -> None:
    async def reconcile(db, artifacts, remote):
        raise AssertionError("startup reconciliation already ran in Env.create")

    client = _Search(ready=ready)
    monkeypatch.setattr(lifespan.index, "reconcile", reconcile)
    env = SimpleNamespace(search=client, db=object(), artifacts=object())

    task = asyncio.create_task(search_reconcile_probe(env, 30.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert client.calls == 0
