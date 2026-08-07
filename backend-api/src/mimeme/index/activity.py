from __future__ import annotations

import asyncio
from typing import Protocol

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
    Build,
    Prepared,
    PrepareInput,
    Result,
    Sealed,
    SealInput,
)
from mimeme.job import ops as job_ops


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
        try:
            activity.logger.info("index prepare started", extra={"job_id": input.job_id})
            result = await index.prepare(self._env.db, self._env.settings, input)
            activity.logger.info(
                "index prepare finished",
                extra={"job_id": result.job_id, "decision": result.decision},
            )
            return result
        except BaseException as failure:
            terminal = _terminal(failure)
            if input.job_id is not None and (
                terminal or activity.info().attempt >= rule.PREPARE_MAX_ATTEMPTS
            ):
                await _bookkeep_failure(self._env, input.job_id, failure, version=None)
            _raise_terminal(failure, terminal)
            raise

    @activity.defn(name=rule.SEAL_ACTIVITY)
    async def seal(self, input: SealInput) -> Sealed:
        heartbeat = asyncio.create_task(_seal_heartbeats(input.job_id))
        try:
            activity.logger.info("index seal started", extra={"job_id": input.job_id})
            result = await index.seal(self._env.db, self._env.index, self._env.settings, input)
            activity.logger.info(
                "index.seal.done",
                extra={
                    "job_id": input.job_id,
                    "model": result.model,
                    "shards": result.shards,
                    "rows": result.rows,
                },
            )
            return result
        except pack.Busy as busy:
            activity.logger.info(
                "index seal skipped", extra={"job_id": input.job_id, "reason": str(busy)}
            )
            return Sealed(model=input.model, shards=0, rows=0)
        except BaseException as failure:
            activity.logger.error(
                "index seal failed",
                extra={"job_id": input.job_id, "error": str(failure)},
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
    async def build(self, request: Build) -> Result:
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
            activity.logger.info(
                "index build started",
                extra={"job_id": request.job_id, "version": request.version},
            )
            result = await index.build(self._env.index, request, progress=progress)
            activity.logger.info(
                "index build finished",
                extra={"job_id": request.job_id, "outcome": result.outcome},
            )
            return result
        except BaseException as failure:
            activity.logger.error(
                "index build failed",
                extra={"job_id": request.job_id, "error": str(failure)},
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
        heartbeat = asyncio.create_task(_activation_heartbeats(input.job_id))
        try:
            activity.logger.info("index activation started", extra={"job_id": input.job_id})
            result = await index.activate(
                self._env.db,
                self._env.artifacts,
                self._env.search,
                input,
                retain=self._env.settings.index.retain_versions,
            )
            activity.logger.info(
                "index activation finished",
                extra={"job_id": input.job_id, "version": result.version},
            )
            return result
        except BaseException as failure:
            activity.logger.error(
                "index activation failed",
                extra={"job_id": input.job_id, "error": str(failure)},
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
