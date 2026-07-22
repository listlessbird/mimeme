from __future__ import annotations

import asyncio

import pytest
from temporalio.testing import ActivityEnvironment

from mimeme import inference
from mimeme.inference.activity import ANNOTATE, EMBED, InferenceActivities
from mimeme.inference.client import Progress


class FakeClient:
    def __init__(self) -> None:
        self.block: asyncio.Event | None = None
        self.cancelled = False

    async def annotate(
        self, input: inference.Input, *, progress: Progress | None = None
    ) -> inference.Annotation:
        if progress is not None:
            await progress("annotate", 0.5)
        if self.block is not None:
            try:
                await self.block.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return inference.Annotation(
            image_id=input.image_id, caption="c", caption_model="m", ocr_text="o", ocr_model="m"
        )

    async def embed(
        self, batch: inference.Batch, *, progress: Progress | None = None
    ) -> inference.BatchResult:
        return inference.BatchResult(items=[])

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def test_registration_names() -> None:
    assert ANNOTATE == "mimeme.inference.annotate.tmp"
    assert EMBED == "mimeme.inference.embed.tmp"


async def test_annotate_activity_returns_and_heartbeats() -> None:
    env = ActivityEnvironment()
    beats: list[tuple] = []
    env.on_heartbeat = lambda *details: beats.append(details)
    activities = InferenceActivities(FakeClient(), poll_interval_s=0.01)
    result = await env.run(activities.annotate, inference.Input(image_id=9, media_key="k"))
    assert result.image_id == 9 and result.caption == "c"
    assert any(d and d[0] == "annotate" for d in beats)


async def test_embed_activity_returns() -> None:
    env = ActivityEnvironment()
    activities = InferenceActivities(FakeClient(), poll_interval_s=0.01)
    result = await env.run(
        activities.embed,
        inference.Batch(items=[]),
    )
    assert result.items == []


async def test_cancellation_propagates_to_client() -> None:
    env = ActivityEnvironment()
    client = FakeClient()
    client.block = asyncio.Event()
    activities = InferenceActivities(client, poll_interval_s=0.01)

    task = asyncio.ensure_future(
        env.run(activities.annotate, inference.Input(image_id=1, media_key="k"))
    )
    await asyncio.sleep(0.05)
    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.cancelled is True
