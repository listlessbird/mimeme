from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from mimeme import release
from mimeme.inference.modal import Modal
from mimeme.inference.model import ANNOTATION_CONTRACT_VERSION, Batch, Context, Input, Item
from mimeme.modal_app.app import VisionService


class FakeCall:
    def __init__(self, result: dict | None = None, block: asyncio.Event | None = None) -> None:
        self._result = result
        self._block = block
        self.cancelled = False
        self.kwargs: dict = {}
        self.get = SimpleNamespace(aio=self._get)
        self.cancel = SimpleNamespace(aio=self._cancel)

    async def _get(self) -> dict:
        if self._block is not None:
            await self._block.wait()
        assert self._result is not None
        return self._result

    async def _cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled = True


class FakeMethod:
    def __init__(self, call: FakeCall) -> None:
        self._call = call
        self.spawn = SimpleNamespace(aio=self._spawn)

    async def _spawn(self, **kwargs: object) -> FakeCall:
        self._call.kwargs = dict(kwargs)
        return self._call


class FakeRemoteFunction:
    def __init__(self, result: dict[str, str | int]) -> None:
        self._result = result
        self.remote = SimpleNamespace(aio=self._call)

    async def _call(self) -> dict[str, str | int]:
        return self._result


def _vision(call: FakeCall):  # noqa: ANN202
    method = FakeMethod(call)
    return lambda: SimpleNamespace(annotate_image=method)


def _embedding(call: FakeCall):  # noqa: ANN202
    method = FakeMethod(call)
    return lambda: SimpleNamespace(embed_batch=method)


def _adapter() -> Modal:
    return Modal(app_name="app", embed_model="google/siglip2", poll_interval_s=0.01)


async def test_annotate_maps_remote_dict() -> None:
    call = FakeCall(
        result={"caption": "c", "caption_model": "m", "ocr_text": "o", "ocr_model": "m"}
    )
    adapter = _adapter()
    adapter._vision = _vision(call)  # type: ignore[assignment]
    result = await adapter.annotate(Input(image_id=3, media_key="k"))
    assert result.image_id == 3 and result.caption == "c"
    assert call.kwargs == {"media_key": "k", "length": "normal"}


async def test_context_matches_the_deployed_annotation_contract() -> None:
    call = FakeCall(
        result={"caption": "c", "caption_model": "m", "ocr_text": "o", "ocr_model": "m"}
    )
    adapter = _adapter()
    adapter._vision = _vision(call)  # type: ignore[assignment]

    await adapter.annotate(
        Input(image_id=3, media_key="k", context=Context(title="Distracted Boyfriend"))
    )

    assert call.kwargs["context"] == {
        "title": "Distracted Boyfriend",
        "description": None,
        "tags": [],
        "categories": [],
        "types": [],
        "origin": None,
        "year": None,
    }
    method = VisionService._get_partial_functions()["annotate_image"]
    inspect.signature(method._get_raw_f()).bind(None, **call.kwargs)


async def test_ready_requires_the_matching_release_and_contract() -> None:
    adapter = _adapter()
    adapter._vision = lambda: None
    adapter._embedding = lambda: None
    adapter._release_info = FakeRemoteFunction(
        {
            "release_id": release.ID,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        }
    )

    assert await adapter.ready() is True

    adapter._release_info = FakeRemoteFunction(
        {
            "release_id": "stale-release",
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        }
    )
    assert await adapter.ready() is False


async def test_embed_builds_keys_and_maps() -> None:
    call = FakeCall(
        result={
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
        }
    )
    adapter = _adapter()
    adapter._embedding = _embedding(call)  # type: ignore[assignment]
    batch = Batch(
        items=[
            Item(image_id=1, media_key="images/1.jpg", sha256="abc", dataset="ds"),
            Item(image_id=2, media_key="images/2.jpg", sha256="def"),
        ]
    )
    result = await adapter.embed(batch)
    assert [e.image_id for e in result.results] == [1]
    assert result.failed_ids == [2]
    sent = {item["image_id"]: item for item in call.kwargs["items"]}
    assert sent[1]["image_key"] == "embeddings/google_siglip2/ds/abc.npy"
    assert "text_key" not in sent[1]


async def test_cancellation_cancels_remote_call() -> None:
    block = asyncio.Event()
    call = FakeCall(result={"caption": "c"}, block=block)
    adapter = _adapter()
    adapter._vision = _vision(call)  # type: ignore[assignment]
    task = asyncio.ensure_future(adapter.annotate(Input(image_id=1, media_key="k")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert call.cancelled is True
