from __future__ import annotations

import asyncio
from typing import Protocol

from temporalio import activity

from mimeme import index, search, storage
from mimeme.db import Db
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
        return await index.prepare(self._env.db, self._env.artifacts, self._env.settings, input)

    @activity.defn(name=rule.BUILD_ACTIVITY)
    async def build(self, request: Build) -> Result:
        async def progress(phase: str, value: float) -> None:
            activity.heartbeat({"phase": phase, "progress": value, "version": request.version})
            if activity.is_cancelled():
                raise asyncio.CancelledError

        try:
            return await index.build(self._env.index, request, progress=progress)
        except BaseException as failure:
            try:
                if isinstance(failure, asyncio.CancelledError):
                    await job_ops.mark_cancelled(self._env.db, request.job_id)
                else:
                    await job_ops.fail_rebuild(self._env.db, request.job_id, str(failure))
                await job_ops.release_claim(self._env.db, request.job_id)
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
        return await index.activate(
            self._env.db,
            self._env.artifacts,
            self._env.search,
            input,
            retain=self._env.settings.index.retain_versions,
        )
