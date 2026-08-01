from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    SchedulePolicy,
    ScheduleState,
    ScheduleUpdate,
    ScheduleUpdateInput,
)
from temporalio.client import ScheduleOverlapPolicy as TemporalOverlap
from temporalio.client import ScheduleSpec as TemporalSpec
from temporalio.service import RPCError, RPCStatusCode

from mimeme.index import rule
from mimeme.index.model import Trigger, WorkflowInput
from mimeme.index.workflow import RebuildWorkflow


class Desired(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"


class Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schedule_id: str = rule.SCHEDULE_ID
    enabled: bool
    cron: str
    timezone: str

    @field_validator("cron", "timezone")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("schedule cron and timezone must not be blank")
        return value

    @property
    def desired(self) -> Desired:
        return Desired.ACTIVE if self.enabled else Desired.PAUSED


class Existing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    exists: bool
    cron: str | None = None
    timezone: str | None = None
    paused: bool = False


def changed(spec: Spec, existing: Existing) -> bool:
    return not existing.exists or (
        existing.cron != spec.cron
        or existing.timezone != spec.timezone
        or existing.paused != (spec.desired is Desired.PAUSED)
    )


class Temporal:
    def __init__(self, client: Client, *, task_queue: str = rule.TASK_QUEUE) -> None:
        self._client = client
        self._queue = task_queue

    async def reconcile(
        self, spec: Spec, *, model: str, index_type: Literal["flat", "hnsw"]
    ) -> bool:
        handle = self._client.get_schedule_handle(spec.schedule_id)
        try:
            description = await handle.describe()
            cron = description.schedule.spec.cron_expressions
            existing = Existing(
                exists=True,
                cron=cron[0] if cron else None,
                timezone=description.schedule.spec.time_zone_name,
                paused=description.schedule.state.paused,
            )
        except RPCError as exc:
            if exc.status is not RPCStatusCode.NOT_FOUND:
                raise
            existing = Existing(exists=False)
        if not changed(spec, existing):
            return False
        schedule = self._schedule(spec, model=model, index_type=index_type)
        if not existing.exists:
            await self._client.create_schedule(spec.schedule_id, schedule)
        else:

            async def update(_: ScheduleUpdateInput) -> ScheduleUpdate:
                return ScheduleUpdate(schedule=schedule)

            await handle.update(update)
        return True

    def _schedule(self, spec: Spec, *, model: str, index_type: Literal["flat", "hnsw"]) -> Schedule:
        return Schedule(
            action=ScheduleActionStartWorkflow(
                RebuildWorkflow.run,
                WorkflowInput(
                    job_id=None,
                    model=model,
                    index_type=index_type,
                    trigger=Trigger.SCHEDULED,
                ),
                id=rule.SCHEDULE_ID,
                task_queue=self._queue,
            ),
            spec=TemporalSpec(
                cron_expressions=[spec.cron],
                time_zone_name=spec.timezone,
            ),
            policy=SchedulePolicy(overlap=TemporalOverlap.SKIP),
            state=ScheduleState(paused=spec.desired is Desired.PAUSED),
        )
