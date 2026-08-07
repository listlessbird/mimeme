from __future__ import annotations

import asyncio
from contextlib import suppress

from temporalio import activity

from mimeme.ingest import rule
from mimeme.ingest.model import Finish, Input, Result
from mimeme.ingest.run import Deps, finish, run
from mimeme.ingest.store import Store
from mimeme.job import ops as job_ops
from mimeme.job.model import IngestResult


class IngestActivities:
    def __init__(self, env: Deps, *, poll_interval_s: float = 5.0) -> None:
        self._env = env
        self._poll = poll_interval_s

    @activity.defn(name=rule.ITEM_ACTIVITY)
    async def item(self, input: Input) -> Result:
        result = await self._drive(input)
        await self._update_progress(input.job_id)
        return result

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(self, input: Finish) -> IngestResult:
        processed, failed, duplicates = await finish(self._env, input.job_id)
        return IngestResult(processed=processed, failed=failed, duplicates=duplicates)

    async def _drive(self, input: Input) -> Result:
        task = asyncio.ensure_future(run(self._env, input))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._poll)
                if task in done:
                    return task.result()
                activity.heartbeat("processing")
                if activity.is_cancelled():
                    await _abort(task)
                    raise asyncio.CancelledError
        except asyncio.CancelledError:
            await _abort(task)
            raise

    async def _update_progress(self, job_id: str) -> None:
        async with self._env.db.read_session() as session:
            completed, total = await Store(session).progress_counts(job_id)
        await job_ops.progress(self._env.db, job_id, rule.progress_percent(completed, total))


async def _abort(task: asyncio.Future) -> None:
    if not task.done():
        task.cancel()
    with suppress(BaseException):
        await task
