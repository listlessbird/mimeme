from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeVar

from temporalio import activity

from mimeme import inference
from mimeme.inference.client import Client, Progress

ANNOTATE = "mimeme.inference.annotate.tmp"
EMBED = "mimeme.inference.embed.tmp"

_T = TypeVar("_T")


class InferenceActivities:
    def __init__(self, client: Client, *, poll_interval_s: float = 5.0) -> None:
        self._client = client
        self._poll = poll_interval_s

    @activity.defn(name=ANNOTATE)
    async def annotate(self, input: inference.Input) -> inference.Annotation:
        return await self._drive(lambda progress: self._client.annotate(input, progress=progress))

    @activity.defn(name=EMBED)
    async def embed(self, batch: inference.Batch) -> inference.BatchResult:
        return await self._drive(lambda progress: self._client.embed(batch, progress=progress))

    async def _drive(self, call: Callable[[Progress], Awaitable[_T]]) -> _T:
        async def progress(phase: str, fraction: float) -> None:
            activity.heartbeat(phase, fraction)

        task = asyncio.ensure_future(call(progress))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._poll)
                if task in done:
                    return task.result()
                activity.heartbeat("waiting", 0.0)
                if activity.is_cancelled():
                    await _abort(task)
                    raise asyncio.CancelledError
        except asyncio.CancelledError:
            await _abort(task)
            raise


async def _abort(task: asyncio.Future) -> None:
    if not task.done():
        task.cancel()
    with suppress(BaseException):
        await task
