from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from datetime import UTC, datetime

import structlog
from temporalio import activity

from mimeme.ingest import rule
from mimeme.ingest.model import BatchInput, Finish, Input, Result
from mimeme.ingest.run import Deps, finish, run, run_batch
from mimeme.ingest.store import Store
from mimeme.job import ops as job_ops
from mimeme.job.model import IngestResult
from mimeme.logging import emit_activity_event


class IngestActivities:
    def __init__(self, env: Deps, *, poll_interval_s: float = 5.0) -> None:
        self._env = env
        self._poll = poll_interval_s

    @activity.defn(name=rule.ITEM_ACTIVITY)
    async def item(self, input: Input) -> Result:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            result = await self._drive(input)
            await self._update_progress(input.job_id)
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="ingest_item_attempt_failed",
                activity_name=rule.ITEM_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                job_id=input.job_id,
                item_id=input.item_id,
                ingest_url_id=input.item_id,
            )
            raise
        _emit_completed(log, rule.ITEM_ACTIVITY, input.job_id, result, started)
        return result

    @activity.defn(name=rule.BATCH_ACTIVITY)
    async def batch(self, input: BatchInput) -> list[Result]:
        started = time.monotonic()
        log = structlog.get_logger()
        task = asyncio.ensure_future(run_batch(self._env, input.items))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._poll)
                if task in done:
                    results = task.result()
                    await self._update_progress(input.items[0].job_id)
                    for result in results:
                        _emit_completed(
                            log, rule.BATCH_ACTIVITY, input.items[0].job_id, result, started
                        )
                    return results
                activity.heartbeat("processing")
                if activity.is_cancelled():
                    await _abort(task)
                    raise asyncio.CancelledError
        except asyncio.CancelledError:
            await _abort(task)
            raise
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="ingest_batch_attempt_failed",
                activity_name=rule.BATCH_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                job_id=input.items[0].job_id,
            )
            raise

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(self, input: Finish) -> IngestResult:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            job = await job_ops.find(self._env.db, input.job_id)
            processed, failed, duplicates = await finish(self._env, input.job_id)
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="ingest_job_attempt_failed",
                activity_name=rule.FINISH_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                job_id=input.job_id,
            )
            raise
        result = IngestResult(processed=processed, failed=failed, duplicates=duplicates)
        job_duration_ms = (
            round((datetime.now(UTC) - job.started_at).total_seconds() * 1000, 2)
            if job is not None and job.started_at is not None
            else None
        )
        emit_activity_event(
            log=log,
            event_name="ingest_job_completed",
            activity_name=rule.FINISH_ACTIVITY,
            started_at=started,
            outcome="completed",
            job_id=input.job_id,
            processed=processed,
            duplicates=duplicates,
            failed=failed,
            total=processed + duplicates + failed,
            duration_ms=job_duration_ms,
        )
        return result

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


def _emit_completed(
    log,
    activity_name: str,
    job_id: str,
    result: Result,
    started: float,  # noqa: ANN001
) -> None:
    emit_activity_event(
        log=log,
        event_name="ingest_item_completed",
        activity_name=activity_name,
        started_at=started,
        outcome=result.outcome,
        job_id=job_id,
        item_id=result.item_id,
        ingest_url_id=result.item_id,
        image_id=result.image_id,
        duplicate_reason=(
            result.duplicate_reason.value if result.duplicate_reason is not None else None
        ),
        error=result.error,
        download_ms=result.download_ms,
        annotation_ms=result.annotation_ms,
        embedding_ms=result.embedding_ms,
        total_ms=result.total_ms,
    )
