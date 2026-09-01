from __future__ import annotations

import asyncio
import json
from time import perf_counter

import httpx
import pytest

from mimeme.inference.local import Local
from mimeme.inference.model import Batch, Input, Invalid, Item, Unavailable

BASE = "http://compute:8010"


def _client(handler) -> httpx.AsyncClient:  # noqa: ANN001
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _adapter(handler) -> Local:  # noqa: ANN001
    return Local(
        _client(handler), base_url=BASE, embed_model="google/siglip2", poll_interval_s=0.01
    )


async def test_annotate_drives_and_maps() -> None:
    calls = {"get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        calls["get"] += 1
        if calls["get"] < 2:
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        return httpx.Response(
            200,
            json={
                "job_id": job_id,
                "status": "succeeded",
                "result": {
                    "caption": "a dog",
                    "caption_model": "moon",
                    "ocr_text": "hello",
                    "ocr_model": "moon",
                },
            },
        )

    adapter = _adapter(handler)
    result = await adapter.annotate(Input(image_id=5, media_key="images/a.jpg"))
    assert result.image_id == 5
    assert result.caption == "a dog"
    assert result.ocr_text == "hello"


async def test_completion_does_not_wait_for_the_poll_interval() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        assert request.url.params["wait_s"] == "5.0"
        await asyncio.sleep(0.03)
        return httpx.Response(
            200,
            json={
                "job_id": job_id,
                "status": "succeeded",
                "result": {
                    "caption": "done",
                    "caption_model": "m",
                    "ocr_text": "",
                    "ocr_model": "m",
                },
            },
        )

    adapter = Local(
        _client(handler), base_url=BASE, embed_model="google/siglip2", poll_interval_s=5.0
    )
    started = perf_counter()
    await adapter.annotate(Input(image_id=1, media_key="images/a.jpg"))

    assert perf_counter() - started < 0.5


async def test_embed_builds_keys_and_maps() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            seen["spec"] = json.loads(request.content)
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        return httpx.Response(
            200,
            json={
                "job_id": job_id,
                "status": "succeeded",
                "result": {
                    "items": [
                        {
                            "image_id": 1,
                            "ok": True,
                            "image_key": "e/1.npy",
                            "model": "google/siglip2",
                            "dimension": 768,
                        },
                        {"image_id": 2, "ok": False, "error": "bad"},
                    ]
                },
            },
        )

    adapter = _adapter(handler)
    batch = Batch(
        items=[
            Item(image_id=1, media_key="images/1.jpg", sha256="abc", dataset="ds"),
            Item(image_id=2, media_key="images/2.jpg", sha256="def"),
        ]
    )
    result = await adapter.embed(batch)
    assert [e.image_id for e in result.results] == [1]
    assert result.failed_ids == [2]
    spec = seen["spec"]
    assert isinstance(spec, dict)
    assert spec["items"][0]["image_key"] == "embeddings/google_siglip2/ds/abc.npy"
    assert "text_key" not in spec["items"][0]


async def test_failed_job_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        return httpx.Response(
            200, json={"job_id": job_id, "status": "failed", "error": "child_dead: boom"}
        )

    adapter = _adapter(handler)
    with pytest.raises(Unavailable, match="child_dead"):
        await adapter.annotate(Input(image_id=1, media_key="k"))


async def test_invalid_inference_output_remains_terminal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        return httpx.Response(
            200, json={"job_id": job_id, "status": "failed", "error": "ValueError: bad output"}
        )

    adapter = _adapter(handler)
    with pytest.raises(Invalid, match="bad output"):
        await adapter.annotate(Input(image_id=1, media_key="k"))


async def test_cancellation_sends_delete() -> None:
    deleted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "DELETE":
            deleted.set()
            return httpx.Response(200, json={"job_id": job_id, "status": "cancelled"})
        return httpx.Response(200, json={"job_id": job_id, "status": "running"})

    adapter = _adapter(handler)
    task = asyncio.ensure_future(adapter.annotate(Input(image_id=1, media_key="k")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert deleted.is_set()


async def test_malformed_response_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not": "a job state"})

    adapter = _adapter(handler)
    with pytest.raises(Unavailable):
        await adapter.annotate(Input(image_id=1, media_key="k"))


async def test_loop_progresses_during_slow_compute() -> None:
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        polls["n"] += 1
        if polls["n"] < 4:
            return httpx.Response(200, json={"job_id": job_id, "status": "running"})
        return httpx.Response(
            200,
            json={
                "job_id": job_id,
                "status": "succeeded",
                "result": {
                    "caption": "c",
                    "caption_model": "m",
                    "ocr_text": "",
                    "ocr_model": "m",
                },
            },
        )

    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    background = asyncio.ensure_future(ticker())
    adapter = _adapter(handler)
    await adapter.annotate(Input(image_id=1, media_key="k"))
    stop.set()
    await background
    assert ticks > 5


async def test_ready_uses_inference_role_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/roles/inference/ready"
        return httpx.Response(200, json={"ok": True, "roles": []})

    adapter = _adapter(handler)
    assert await adapter.ready() is True
