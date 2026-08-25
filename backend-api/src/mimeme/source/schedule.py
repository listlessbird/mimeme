from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    SchedulePolicy,
    ScheduleState,
    ScheduleUpdate,
    ScheduleUpdateInput,
)
from temporalio.client import ScheduleOverlapPolicy as TemporalScheduleOverlap
from temporalio.client import ScheduleSpec as TemporalScheduleSpec

from mimeme.db.schema import SourceRunTrigger
from mimeme.source import rule


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ScheduleOverlapPolicy(StrEnum):
    SKIP = "skip"


class DesiredScheduleState(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ABSENT = "absent"


class ScheduleSpec(_Frozen):
    source_id: int
    schedule_id: str
    cron: str | None
    timezone: str
    overlap_policy: ScheduleOverlapPolicy
    desired_state: DesiredScheduleState


class ExistingSchedule(_Frozen):
    schedule_id: str
    cron: str | None
    timezone: str | None
    paused: bool


class CreateSchedule(_Frozen):
    spec: ScheduleSpec


class UpdateSchedule(_Frozen):
    spec: ScheduleSpec


class DeleteSchedule(_Frozen):
    schedule_id: str


ScheduleAction = CreateSchedule | UpdateSchedule | DeleteSchedule


def derive_schedule_spec(
    *,
    source_id: int,
    schedule_cron: str | None,
    schedule_timezone: str,
    enabled: bool,
    deleted: bool,
) -> ScheduleSpec:
    if deleted or schedule_cron is None:
        desired_state = DesiredScheduleState.ABSENT
    elif not enabled:
        desired_state = DesiredScheduleState.PAUSED
    else:
        desired_state = DesiredScheduleState.ACTIVE

    return ScheduleSpec(
        source_id=source_id,
        schedule_id=rule.schedule_id(source_id),
        cron=schedule_cron,
        timezone=schedule_timezone,
        overlap_policy=ScheduleOverlapPolicy.SKIP,
        desired_state=desired_state,
    )


def plan_reconciliation(
    *, desired: list[ScheduleSpec], existing: list[ExistingSchedule]
) -> list[ScheduleAction]:
    existing_by_id = {schedule.schedule_id: schedule for schedule in existing}
    desired_ids = {spec.schedule_id for spec in desired}

    actions: list[ScheduleAction] = []

    for spec in desired:
        current = existing_by_id.get(spec.schedule_id)

        if spec.desired_state == DesiredScheduleState.ABSENT:
            if current is not None:
                actions.append(DeleteSchedule(schedule_id=spec.schedule_id))
            continue

        if current is None:
            actions.append(CreateSchedule(spec=spec))
            continue

        wanted_pause = spec.desired_state == DesiredScheduleState.PAUSED
        if (
            current.cron == spec.cron
            and current.paused == wanted_pause
            and current.timezone == spec.timezone
        ):
            continue

        actions.append(UpdateSchedule(spec=spec))

    for schedule in existing:
        if schedule.schedule_id not in desired_ids:
            actions.append(DeleteSchedule(schedule_id=schedule.schedule_id))

    return actions


class ScheduleStore(Protocol):
    async def list_existing(self) -> list[ExistingSchedule]: ...
    async def create(self, spec: ScheduleSpec) -> None: ...
    async def update(self, spec: ScheduleSpec) -> None: ...
    async def delete(self, schedule_id: str) -> None: ...


async def reconcile(store: ScheduleStore, *, desired: list[ScheduleSpec]) -> list[ScheduleAction]:
    existing = await store.list_existing()
    plan = plan_reconciliation(desired=desired, existing=existing)

    for action in plan:
        match action:
            case CreateSchedule(spec=spec):
                await store.create(spec)
            case UpdateSchedule(spec=spec):
                await store.update(spec)
            case DeleteSchedule(schedule_id=schedule_id):
                await store.delete(schedule_id)

    return plan


class TemporalScheduleStore:
    def __init__(self, client: Client, *, task_queue: str) -> None:
        self._client = client
        self._task_queue = task_queue

    async def list_existing(self) -> list[ExistingSchedule]:
        existing: list[ExistingSchedule] = []
        async for listed in await self._client.list_schedules():
            if not listed.id.startswith(rule.SCHEDULE_PREFIX):
                continue
            desc = await self._client.get_schedule_handle(listed.id).describe()
            cron = desc.schedule.spec.cron_expressions
            existing.append(
                ExistingSchedule(
                    schedule_id=listed.id,
                    cron=cron[0] if cron else None,
                    timezone=desc.schedule.spec.time_zone_name or None,
                    paused=desc.schedule.state.paused,
                )
            )
        return existing

    async def create(self, spec: ScheduleSpec) -> None:
        await self._client.create_schedule(spec.schedule_id, self._build_schedule(spec))

    async def delete(self, schedule_id: str) -> None:
        await self._client.get_schedule_handle(schedule_id).delete()

    async def update(self, spec: ScheduleSpec) -> None:
        handle = self._client.get_schedule_handle(spec.schedule_id)

        async def updater(update_input: ScheduleUpdateInput) -> ScheduleUpdate:
            schedule = update_input.description.schedule
            schedule.spec = TemporalScheduleSpec(
                cron_expressions=[_cron_required(spec)], time_zone_name=spec.timezone
            )
            schedule.policy = SchedulePolicy(overlap=TemporalScheduleOverlap.SKIP)
            schedule.state = ScheduleState(paused=spec.desired_state == DesiredScheduleState.PAUSED)
            return ScheduleUpdate(schedule=schedule)

        await handle.update(updater)

    def _build_schedule(self, spec: ScheduleSpec) -> Schedule:
        from mimeme.source.model import SyncInput
        from mimeme.source.workflow import SourceSyncWorkflow

        return Schedule(
            action=ScheduleActionStartWorkflow(
                SourceSyncWorkflow.run,
                SyncInput(source_id=spec.source_id, trigger=SourceRunTrigger.SCHEDULED),
                id=spec.schedule_id,
                task_queue=self._task_queue,
            ),
            spec=TemporalScheduleSpec(
                cron_expressions=[_cron_required(spec)], time_zone_name=spec.timezone
            ),
            policy=SchedulePolicy(overlap=TemporalScheduleOverlap.SKIP),
            state=ScheduleState(paused=spec.desired_state == DesiredScheduleState.PAUSED),
        )


def _cron_required(spec: ScheduleSpec) -> str:
    if spec.cron is None:
        raise ValueError(
            f"Schedule {spec.schedule_id} cannot be created/updated without a cron exp"
        )
    return spec.cron
