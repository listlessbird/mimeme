from __future__ import annotations

import asyncio

import httpx
import pytest

from mimeme import index
from mimeme.index.local import Local


def _build() -> index.Build:
    return index.Build(
        job_id="rebuild-1",
        version="v2-g1-test",
        target_generation=1,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        embeddings=[index.Embedding(image_id=1, image_key="embeddings/1.npy")],
    )


def test_local_compute_poll_interval_is_never_more_than_five_seconds() -> None:
    client = Local(httpx.AsyncClient(), base_url="http://compute", poll_interval_s=60)
    assert client._poll == 5.0  # noqa: SLF001


async def test_cancellation_propagates_to_the_stable_compute_job() -> None:
    polling = asyncio.Event()
    deleted: list[str] = []

    class Http:
        async def put(self, url: str, *, json: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={"job_id": "rebuild-1", "status": "running", "progress": 0},
                request=httpx.Request("PUT", url),
            )

        async def get(self, url: str) -> httpx.Response:
            polling.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def delete(self, url: str) -> httpx.Response:
            deleted.append(url)
            return httpx.Response(200, request=httpx.Request("DELETE", url))

    client = Local(Http(), base_url="http://compute", poll_interval_s=0.01)  # type: ignore[arg-type]
    task = asyncio.create_task(client.build(_build()))
    await polling.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert deleted == ["http://compute/v1/jobs/rebuild-1"]
