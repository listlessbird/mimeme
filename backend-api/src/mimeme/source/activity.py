from __future__ import annotations

from temporalio import activity

from mimeme.source import rule, sync
from mimeme.source.model import DiscoverInput, DiscoverResult, FinishInput, FinishResult


class SourceActivities:
    def __init__(self, env: sync.Deps) -> None:
        self._env = env

    @activity.defn(name=rule.DISCOVER_ACTIVITY)
    async def discover(self, input: DiscoverInput) -> DiscoverResult:
        return await sync.discover(
            self._env,
            input,
            heartbeat=lambda message: activity.heartbeat(message),
            cancelled=activity.is_cancelled,
        )

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(self, input: FinishInput) -> FinishResult:
        return await sync.finish(self._env, input)
