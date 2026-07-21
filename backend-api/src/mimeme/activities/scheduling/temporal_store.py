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
from mimeme.domain.source_schedule_reconcile import ExistingSchedule
from mimeme.domain.source_schedule_spec import DesiredScheduleState, ScheduleSpec
from mimeme.shared.runtime import settings
from mimeme.workflows.models import SourceSyncWorkflowInput
from mimeme.workflows.source_sync import SourceSyncWorkflow

_SOURCE_SYNC_PREFIX = "source-sync-"


class TemporalScheduleStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def list_existing(self) -> list[ExistingSchedule]:
        existing: list[ExistingSchedule] = []

        async for listed in await self._client.list_schedules():
            if not listed.id.startswith(_SOURCE_SYNC_PREFIX):
                continue

            handle = self._client.get_schedule_handle(listed.id)
            desc = await handle.describe()

            cron_exp = desc.schedule.spec.cron_expressions

            existing.append(
                ExistingSchedule(
                    schedule_id=listed.id,
                    cron=cron_exp[0] if cron_exp else None,
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

        async def updater(input: ScheduleUpdateInput) -> ScheduleUpdate:
            schedule = input.description.schedule

            schedule.spec = TemporalScheduleSpec(
                cron_expressions=[_cron_required(spec)], time_zone_name=spec.timezone
            )

            schedule.policy = SchedulePolicy(overlap=TemporalScheduleOverlap.SKIP)

            schedule.state = ScheduleState(paused=spec.desired_state == DesiredScheduleState.PAUSED)

            return ScheduleUpdate(schedule=schedule)

        await handle.update(updater)

    def _build_schedule(self, spec: ScheduleSpec) -> Schedule:
        return Schedule(
            action=ScheduleActionStartWorkflow(
                SourceSyncWorkflow.run,
                SourceSyncWorkflowInput(
                    source_id=spec.source_id, trigger=SourceRunTrigger.SCHEDULED
                ),
                id=spec.schedule_id,
                task_queue=settings.temporal.task_queue,
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
