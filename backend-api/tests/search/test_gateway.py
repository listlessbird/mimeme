from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from mimeme import search
from mimeme.compute.model import ChildOk
from mimeme.compute.supervisor import ChildDead
from mimeme.search.gateway import Gateway
from mimeme.storage import Object
from tests.support.storage import Memory


class _Supervisor:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.dead = False
        self.restarts = 0
        self.candidate: str | None = None
        self.serving: str | None = None
        self.retained: str | None = None

    async def call(self, role: str, request: bytes) -> bytes:
        if self.dead:
            await asyncio.sleep(0)
            raise ChildDead("gone")
        payload = json.loads(request)
        self.calls.append(payload)
        if payload["op"] == "search.load":
            for path in payload["load"]["paths"].values():
                assert Path(path).exists()
            self.candidate = payload["load"]["version"]
            result = {
                "version": self.candidate,
                "embed_model": "test/embed",
                "dimension": 2,
                "image_count": 3,
                "faiss_version": "1.13.2",
                "onnxruntime_version": "1.27.0",
                "encoder_revision": "rev-1",
            }
        else:
            if payload["op"] == "search.switch":
                self.retained = self.serving
                self.serving = self.candidate
                self.candidate = None
            elif payload["op"] == "search.rollback":
                self.candidate = self.serving
                self.serving = self.retained
                self.retained = None
            result = {
                "ready": self.serving is not None,
                "serving_version": self.serving,
                "candidate_version": self.candidate,
                "retained_version": self.retained,
                "embed_model": "test/embed",
                "encoder_repo": "test/encoder",
                "encoder_revision": "rev-1",
                "encoder_variant": "model.onnx",
                "detail": None if self.serving else "no generation",
            }
        return ChildOk(result=result).model_dump_json().encode()

    async def restart(self, role: str) -> None:
        self.restarts += 1
        self.dead = False
        self.candidate = None
        self.serving = None
        self.retained = None


async def _generation(store: Memory, version: str) -> search.Load:
    files: list[search.File] = []
    for name, data in (
        ("index.faiss", f"index-{version}".encode()),
        ("mapping.json", b"{}"),
        ("metadata.json", b"{}"),
    ):
        key = f"indexes/{version}/{name}"
        await store.put_bytes(Object(key), data, content_type="application/octet-stream")
        files.append(search.File(name=name, key=key, sha256=hashlib.sha256(data).hexdigest()))
    return search.Load(
        version=version,
        files=files,
        encoder=search.Encoder(
            repo="test/encoder", revision="rev-1", variant="model.onnx", threads=1
        ),
    )


async def test_gateway_hydrates_and_verifies_generation_before_child_load(
    tmp_path: Path,
) -> None:
    store = Memory()
    supervisor = _Supervisor()
    gateway = Gateway(supervisor, artifacts=store, workspace_dir=tmp_path)

    loaded = await gateway.load(await _generation(store, "v1"))

    assert loaded.version == "v1"
    assert supervisor.calls[0]["op"] == "search.load"
    roots = list(tmp_path.iterdir())
    assert len(roots) == 1
    assert roots[0].is_dir()


async def test_gateway_clear_restarts_search_without_a_serving_generation(tmp_path: Path) -> None:
    store = Memory()
    supervisor = _Supervisor()
    gateway = Gateway(supervisor, artifacts=store, workspace_dir=tmp_path)
    await gateway.load(await _generation(store, "v1"))
    await gateway.switch("v1")

    status = await gateway.clear()

    assert supervisor.restarts == 1
    assert status.serving_version is None
    assert list(tmp_path.iterdir()) == []


async def test_gateway_restarts_a_dead_child_and_returns_bounded_unavailability(
    tmp_path: Path,
) -> None:
    class Dead:
        def __init__(self) -> None:
            self.restarted = False

        async def call(self, role: str, request: bytes) -> bytes:
            raise ChildDead("gone")

        async def restart(self, role: str) -> None:
            self.restarted = True

    supervisor = Dead()
    gateway = Gateway(supervisor, artifacts=Memory(), workspace_dir=tmp_path)

    with pytest.raises(search.Unavailable, match="without a known serving generation"):
        await gateway.status()

    assert supervisor.restarted is True


async def test_gateway_restores_the_serving_generation_after_child_death(
    tmp_path: Path,
) -> None:
    store = Memory()
    supervisor = _Supervisor()
    gateway = Gateway(supervisor, artifacts=store, workspace_dir=tmp_path)
    generation = await _generation(store, "v1")
    await gateway.load(generation)
    await gateway.switch("v1")
    supervisor.dead = True

    with pytest.raises(search.Unavailable, match="restored its serving generation"):
        await gateway.status()

    assert supervisor.restarts == 1
    assert [call["op"] for call in supervisor.calls] == [
        "search.load",
        "search.switch",
        "search.load",
        "search.switch",
    ]


async def test_concurrent_child_failures_share_one_recovery(tmp_path: Path) -> None:
    store = Memory()
    supervisor = _Supervisor()
    gateway = Gateway(supervisor, artifacts=store, workspace_dir=tmp_path)
    await gateway.load(await _generation(store, "v1"))
    await gateway.switch("v1")
    supervisor.dead = True

    results = await asyncio.gather(gateway.status(), gateway.status(), return_exceptions=True)

    assert all(isinstance(result, search.Unavailable) for result in results)
    assert supervisor.restarts == 1


async def test_child_recovery_restores_the_retained_rollback_generation(
    tmp_path: Path,
) -> None:
    store = Memory()
    supervisor = _Supervisor()
    gateway = Gateway(supervisor, artifacts=store, workspace_dir=tmp_path)
    await gateway.load(await _generation(store, "v1"))
    await gateway.switch("v1")
    await gateway.load(await _generation(store, "v2"))
    await gateway.switch("v2")
    supervisor.dead = True

    with pytest.raises(search.Unavailable, match="restored its serving generation"):
        await gateway.status()
    status = await gateway.rollback("v2")

    assert status.serving_version == "v1"
    assert supervisor.restarts == 1
