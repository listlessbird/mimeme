from __future__ import annotations

from typing import Protocol

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
from temporalio.service import RPCError, RPCStatusCode

from mimeme.db.schema import RebuildTrigger
from mimeme.domain.index_rebuild_schedule import (
    REBUILD_SCHEDULE_ID,
    ExistingRebuildSchedule,
    RebuildScheduleAction,
    RebuildScheduleSpec,
    derive_rebuild_schedule_spec,
    plan_rebuild_schedule,
)
from mimeme.shared.runtime import settings
from mimeme.workflows.models import RebuildIndexWorkflowInput
from mimeme.workflows.rebuild_index import RebuildIndexWorkflow


class RebuildScheduleStore(Protocol):
    async def describe(self) -> ExistingRebuildSchedule: ...
    async def create(self, spec: RebuildScheduleSpec) -> None: ...
    async def update(self, spec: RebuildScheduleSpec) -> None: ...
    async def pause(self, schedule_id: str) -> None: ...


async def reconcile_rebuild_schedule(
    store: RebuildScheduleStore, *, spec: RebuildScheduleSpec
) -> RebuildScheduleAction:
    existing = await store.describe()
    action = plan_rebuild_schedule(spec=spec, existing=existing)

    match action:
        case "create":
            await store.create(spec)
        case "update":
            await store.update(spec)
        case "pause":
            await store.pause(spec.schedule_id)
        case "noop":
            pass

    return action


class TemporalRebuildScheduleStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def describe(self) -> ExistingRebuildSchedule:
        try:
            desc = await self._client.get_schedule_handle(REBUILD_SCHEDULE_ID).describe()
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return ExistingRebuildSchedule(exists=False)
            raise

        cron = desc.schedule.spec.cron_expressions
        return ExistingRebuildSchedule(
            exists=True,
            cron=cron[0] if cron else None,
            timezone=desc.schedule.spec.time_zone_name or None,
            paused=desc.schedule.state.paused,
        )

    async def create(self, spec: RebuildScheduleSpec) -> None:
        await self._client.create_schedule(spec.schedule_id, self._build_schedule(spec))

    async def update(self, spec: RebuildScheduleSpec) -> None:
        handle = self._client.get_schedule_handle(spec.schedule_id)

        async def updater(update_input: ScheduleUpdateInput) -> ScheduleUpdate:
            schedule = update_input.description.schedule
            schedule.spec = TemporalScheduleSpec(
                cron_expressions=[spec.cron], time_zone_name=spec.timezone
            )
            schedule.policy = SchedulePolicy(overlap=TemporalScheduleOverlap.SKIP)
            schedule.state = ScheduleState(paused=False)
            return ScheduleUpdate(schedule=schedule)

        await handle.update(updater)

    async def pause(self, schedule_id: str) -> None:
        await self._client.get_schedule_handle(schedule_id).pause()

    def _build_schedule(self, spec: RebuildScheduleSpec) -> Schedule:
        return Schedule(
            action=ScheduleActionStartWorkflow(
                RebuildIndexWorkflow.run,
                RebuildIndexWorkflowInput(
                    job_id=None,
                    force=False,
                    model_name=settings.inference.embed_model,
                    index_type=settings.index.type,
                    trigger=RebuildTrigger.SCHEDULED,
                ),
                id=spec.action_id,
                task_queue=settings.temporal.task_queue,
            ),
            spec=TemporalScheduleSpec(cron_expressions=[spec.cron], time_zone_name=spec.timezone),
            policy=SchedulePolicy(overlap=TemporalScheduleOverlap.SKIP),
            state=ScheduleState(paused=False),
        )


async def run_rebuild_schedule_reconciliation(client: Client) -> RebuildScheduleAction:
    spec = derive_rebuild_schedule_spec(
        enabled=settings.index.rebuild_schedule_enabled,
        cron=settings.index.rebuild_schedule_cron,
        timezone=settings.index.rebuild_schedule_timezone,
    )
    return await reconcile_rebuild_schedule(TemporalRebuildScheduleStore(client), spec=spec)
