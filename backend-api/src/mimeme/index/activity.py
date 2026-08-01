from __future__ import annotations

import asyncio
from typing import Protocol

from temporalio import activity

from mimeme import search, storage
from mimeme.db import Db
from mimeme.index import ops as index
from mimeme.index import rule
from mimeme.index.client import Client
from mimeme.index.model import Activated, ActivateInput, Build, Prepared, PrepareInput, Result
from mimeme.job import ops as job_ops
from mimeme.shared.config import Settings


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
        activity.logger.info("index prepare started", extra={"job_id": input.job_id})
        result = await index.prepare(self._env.db, self._env.artifacts, self._env.settings, input)
        activity.logger.info(
            "index prepare finished",
            extra={"job_id": result.job_id, "decision": result.decision},
        )
        return result

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
            try:
                final_attempt = activity.info().attempt >= rule.BUILD_MAX_ATTEMPTS
                if isinstance(failure, asyncio.CancelledError) or final_attempt:
                    await index.fail(
                        self._env.db,
                        job_id=request.job_id,
                        error=str(failure),
                        cancelled=isinstance(failure, asyncio.CancelledError),
                    )
                    await index.cleanup_incomplete(
                        self._env.artifacts,
                        version=request.version,
                        protect=set(),
                    )
            except Exception:
                pass
            raise

    @activity.defn(name=rule.ACTIVATE_ACTIVITY)
    async def activate(self, input: ActivateInput) -> Activated:
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
            try:
                final_attempt = activity.info().attempt >= rule.ACTIVATE_MAX_ATTEMPTS
                if isinstance(failure, asyncio.CancelledError) or final_attempt:
                    await index.fail(
                        self._env.db,
                        job_id=input.job_id,
                        error=str(failure),
                        cancelled=isinstance(failure, asyncio.CancelledError),
                    )
                    version = (
                        input.result.manifest.version
                        if input.result is not None and input.result.manifest is not None
                        else None
                    )
                    await index.cleanup_incomplete(
                        self._env.artifacts,
                        version=version,
                        protect=set(),
                    )
            except Exception:
                pass
            raise
