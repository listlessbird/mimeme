from pydantic import BaseModel

from mimeme.domain.source_schedule_spec import DesiredScheduleState, ScheduleSpec


class ExistingSchedule(BaseModel, frozen=True):
    schedule_id: str
    cron: str | None
    paused: bool


class CreateSchedule(BaseModel, frozen=True):
    spec: ScheduleSpec


class UpdateSchedule(BaseModel, frozen=True):
    spec: ScheduleSpec


class DeleteSchedule(BaseModel, frozen=True):
    schedule_id: str


ScheduleAction = CreateSchedule | UpdateSchedule | DeleteSchedule


def plan_reconciliation(
    *, desired: list[ScheduleSpec], existing: list[ExistingSchedule]
) -> list[ScheduleAction]:

    existing_by_id = {schedule.schedule_id: schedule for schedule in existing}
    desired_ids = {spec.schedule_id for spec in desired}

    actions: list[ScheduleAction] = []

    for spec in desired:
        existing_schedule = existing_by_id.get(spec.schedule_id)

        if spec.desired_state == DesiredScheduleState.ABSENT:
            if existing_schedule is not None:
                actions.append(DeleteSchedule(schedule_id=spec.schedule_id))
            continue

        if existing_schedule is None:
            actions.append(CreateSchedule(spec=spec))
            continue

        wanted_pause = spec.desired_state == DesiredScheduleState.PAUSED

        if existing_schedule.cron == spec.cron and existing_schedule.paused == wanted_pause:
            continue

        actions.append(UpdateSchedule(spec=spec))

    for schedule in existing:
        if schedule.schedule_id not in desired_ids:
            actions.append(DeleteSchedule(schedule_id=schedule.schedule_id))

    return actions
