from __future__ import annotations

import time

import structlog
from temporalio import activity

from mimeme.logging import emit_activity_event
from mimeme.source import rule, sync
from mimeme.source.model import (
    CleanupInput,
    DiscoverInput,
    DiscoverResult,
    FinishInput,
    FinishResult,
)


class SourceActivities:
    def __init__(self, env: sync.Deps) -> None:
        self._env = env

    @activity.defn(name=rule.DISCOVER_ACTIVITY)
    async def discover(self, input: DiscoverInput) -> DiscoverResult:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            result = await sync.discover(
                self._env,
                input,
                heartbeat=lambda message: activity.heartbeat(message),
                cancelled=activity.is_cancelled,
            )
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="source_discovery_failed",
                activity_name=rule.DISCOVER_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                source_id=input.source_id,
            )
            raise
        emit_activity_event(
            log=log,
            event_name="source_discovery_completed",
            activity_name=rule.DISCOVER_ACTIVITY,
            started_at=started,
            outcome="completed",
            source_id=input.source_id,
            source_run_id=result.source_run_id,
            ingest_job_id=result.ingest_job_id,
            discovered=result.discovered,
            queued=result.queued,
            dataset=result.dataset,
        )
        if result.ingest_job_id is not None:
            log.info(
                "ingest_job_submitted",
                job_id=result.ingest_job_id,
                source_id=input.source_id,
                source_run_id=result.source_run_id,
                total=result.queued,
                queued=result.queued,
                duplicate_inputs=0,
                dataset=result.dataset,
            )
        return result

    @activity.defn(name=rule.CHECKPOINT_CLEANUP_ACTIVITY)
    async def cleanup_checkpoint(self, input: CleanupInput) -> None:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            await sync.cleanup_checkpoint(self._env, input)
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="source_checkpoint_cleanup_failed",
                activity_name=rule.CHECKPOINT_CLEANUP_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                checkpoint_id=input.checkpoint_id,
            )
            raise

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(self, input: FinishInput) -> FinishResult:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            result = await sync.finish(self._env, input)
        except BaseException as exc:
            emit_activity_event(
                log=log,
                event_name="source_run_failed",
                activity_name=rule.FINISH_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(exc),
                source_run_id=input.source_run_id,
            )
            raise
        emit_activity_event(
            log=log,
            event_name="source_run_completed",
            activity_name=rule.FINISH_ACTIVITY,
            started_at=started,
            outcome=result.status.value.lower(),
            source_id=result.source_id,
            source_run_id=input.source_run_id,
            status=result.status.value,
            discovered=result.discovered,
            queued=result.queued,
            duplicates=result.duplicate,
            failed=result.failed,
            total=result.queued,
            duration_ms=result.duration_ms,
            error=input.error,
        )
        return result
