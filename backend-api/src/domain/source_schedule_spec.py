from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ScheduleOverlapPolicy(StrEnum):
    SKIP = "skip"


class DesiredScheduleState(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ABSENT = "absent"


class ScheduleSpec(BaseModel, frozen=True):
    source_id: int
    schedule_id: str
    cron: str | None
    timezone: str
    overlap_policy: ScheduleOverlapPolicy
    desired_state: DesiredScheduleState


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
        schedule_id=f"source-sync-{source_id}",
        cron=schedule_cron,
        timezone=schedule_timezone,
        overlap_policy=ScheduleOverlapPolicy.SKIP,
        desired_state=desired_state,
    )
