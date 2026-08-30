from __future__ import annotations

import asyncio
import time
from typing import Protocol

import structlog
from temporalio import activity
from temporalio.exceptions import ApplicationError

from mimeme import search, storage
from mimeme.config import Settings
from mimeme.db import Db
from mimeme.index import ops as index
from mimeme.index import pack, rule
from mimeme.index.client import Client
from mimeme.index.model import (
    Activated,
    ActivateInput,
    BuildPlan,
    Prepared,
    PrepareInput,
    Result,
    Sealed,
    SealInput,
)
from mimeme.job import ops as job_ops
from mimeme.logging import emit_activity_event


class Deps(Protocol):
    db: Db
    artifacts: storage.Store
    settings: Settings
    index: Client
    search: search.Activation


class Activities:
    def __init__(self, env: Deps) -> None:
        self._env = env

    @activity.defn(name=rule.PREPARE_ACTIVITY)
    async def prepare(self, input: PrepareInput) -> Prepared:
        started = time.monotonic()
        log = structlog.get_logger()
        try:
            result = await index.prepare(
                self._env.db, self._env.artifacts, self._env.settings, input
            )
            emit_activity_event(
                log=log,
                event_name="index_activity_completed",
                activity_name=rule.PREPARE_ACTIVITY,
                started_at=started,
                outcome=result.decision,
                job_id=result.job_id or input.job_id,
                decision=result.decision,
            )
            return result
        except BaseException as failure:
            emit_activity_event(
                log=log,
                event_name="index_activity_failed",
                activity_name=rule.PREPARE_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(failure),
                job_id=input.job_id,
            )
            terminal = _terminal(failure)
            if input.job_id is not None and (
                terminal or activity.info().attempt >= rule.PREPARE_MAX_ATTEMPTS
            ):
                await _bookkeep_failure(self._env, input.job_id, failure, version=None)
            _raise_terminal(failure, terminal)
            raise

    @activity.defn(name=rule.SEAL_ACTIVITY)
    async def seal(self, input: SealInput) -> Sealed:
        started = time.monotonic()
        log = structlog.get_logger()
        heartbeat = asyncio.create_task(_seal_heartbeats(input.job_id))
        try:
            result = await index.seal(self._env.db, self._env.index, self._env.settings, input)
            emit_activity_event(
                log=log,
                event_name="index_activity_completed",
                activity_name=rule.SEAL_ACTIVITY,
                started_at=started,
                outcome="completed",
                job_id=input.job_id,
                model=result.model,
                shards=result.shards,
                rows=result.rows,
            )
            return result
        except pack.Busy as busy:
            emit_activity_event(
                log=log,
                event_name="index_activity_completed",
                activity_name=rule.SEAL_ACTIVITY,
                started_at=started,
                outcome="skipped",
                job_id=input.job_id,
                reason=str(busy),
            )
            return Sealed(model=input.model, shards=0, rows=0)
        except BaseException as failure:
            emit_activity_event(
                log=log,
                event_name="index_activity_failed",
                activity_name=rule.SEAL_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(failure),
                job_id=input.job_id,
            )
            terminal = _terminal(failure)
            if (
                isinstance(failure, asyncio.CancelledError)
                or terminal
                or activity.info().attempt >= rule.SEAL_MAX_ATTEMPTS
            ):
                await _bookkeep_failure(self._env, input.job_id, failure, version=None)
            _raise_terminal(failure, terminal)
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    @activity.defn(name=rule.BUILD_ACTIVITY)
    async def build(self, request: BuildPlan) -> Result:
        started = time.monotonic()
        log = structlog.get_logger()

        async def progress(phase: str, value: float) -> None:
            activity.heartbeat({"phase": phase, "progress": value, "version": request.version})
            try:
                await job_ops.progress(
                    self._env.db,
                    request.job_id,
                    10 + value * 60,
                    f"Index build: {phase}",
                )
            except Exception:
                pass
            if activity.is_cancelled():
                raise asyncio.CancelledError

        try:
            build = await index.load_build(self._env.artifacts, request)
            result = await index.build(self._env.index, build, progress=progress)
            emit_activity_event(
                log=log,
                event_name="index_activity_completed",
                activity_name=rule.BUILD_ACTIVITY,
                started_at=started,
                outcome=result.outcome,
                job_id=request.job_id,
                version=request.version,
                num_vectors=(result.manifest.image_count if result.manifest else 0),
                dimension=(result.manifest.dimension if result.manifest else 0),
            )
            return result
        except BaseException as failure:
            emit_activity_event(
                log=log,
                event_name="index_activity_failed",
                activity_name=rule.BUILD_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(failure),
                job_id=request.job_id,
                version=request.version,
            )
            terminal = _terminal(failure)
            if (
                isinstance(failure, asyncio.CancelledError)
                or terminal
                or activity.info().attempt >= rule.BUILD_MAX_ATTEMPTS
            ):
                await _bookkeep_failure(self._env, request.job_id, failure, version=request.version)
            _raise_terminal(failure, terminal)
            raise

    @activity.defn(name=rule.ACTIVATE_ACTIVITY)
    async def activate(self, input: ActivateInput) -> Activated:
        started = time.monotonic()
        log = structlog.get_logger()
        heartbeat = asyncio.create_task(_activation_heartbeats(input.job_id))
        try:
            result = await index.activate(
                self._env.db,
                self._env.artifacts,
                self._env.search,
                input,
                retain=self._env.settings.index.retain_versions,
            )
            emit_activity_event(
                log=log,
                event_name="index_activity_completed",
                activity_name=rule.ACTIVATE_ACTIVITY,
                started_at=started,
                outcome="completed",
                job_id=input.job_id,
                version=result.version,
                removed_versions=result.removed_versions,
            )
            return result
        except BaseException as failure:
            emit_activity_event(
                log=log,
                event_name="index_activity_failed",
                activity_name=rule.ACTIVATE_ACTIVITY,
                started_at=started,
                outcome="failed",
                error=str(failure),
                job_id=input.job_id,
            )
            terminal = _terminal(failure)
            if (
                isinstance(failure, asyncio.CancelledError)
                or terminal
                or activity.info().attempt >= rule.ACTIVATE_MAX_ATTEMPTS
            ):
                version = (
                    input.result.manifest.version
                    if input.result is not None and input.result.manifest is not None
                    else None
                )
                await _bookkeep_failure(self._env, input.job_id, failure, version=version)
            _raise_terminal(failure, terminal)
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass


def _terminal(failure: BaseException) -> bool:
    return isinstance(
        failure,
        (
            ValueError,
            storage.Invalid,
            storage.Integrity,
            storage.Missing,
            storage.Denied,
            search.Invalid,
            search.Incompatible,
        ),
    )


def _raise_terminal(failure: BaseException, terminal: bool) -> None:
    if terminal:
        raise ApplicationError(
            str(failure), type=type(failure).__name__, non_retryable=True
        ) from failure


async def _bookkeep_failure(
    env: Deps, job_id: str, failure: BaseException, *, version: str | None
) -> None:
    async def apply() -> None:
        failure_error: Exception | None = None
        try:
            await index.fail(
                env.db,
                job_id=job_id,
                error=str(failure),
                cancelled=isinstance(failure, asyncio.CancelledError),
            )
        except Exception as exc:
            failure_error = exc
        try:
            await index.cleanup_incomplete(env.artifacts, version=version, protect=set())
        except Exception as exc:
            failure_error = failure_error or exc
        if failure_error is not None:
            raise failure_error

    try:
        await asyncio.shield(apply())
    except Exception as bookkeeping:
        raise ApplicationError(
            f"bookkeeping failed after {type(failure).__name__}: {bookkeeping}",
            type="IndexFailureBookkeeping",
        ) from failure


async def _seal_heartbeats(job_id: str) -> None:
    while True:
        activity.heartbeat({"phase": "seal", "job_id": job_id})
        if activity.is_cancelled():
            raise asyncio.CancelledError
        await asyncio.sleep(rule.POLL_INTERVAL_S)


async def _activation_heartbeats(job_id: str) -> None:
    while True:
        activity.heartbeat({"phase": "activate", "job_id": job_id})
        if activity.is_cancelled():
            raise asyncio.CancelledError
        await asyncio.sleep(rule.POLL_INTERVAL_S)
