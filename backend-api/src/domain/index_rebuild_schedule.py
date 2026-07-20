from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

REBUILD_SCHEDULE_ID = "search-index-rebuild"
REBUILD_ACTION_ID = "search-index-rebuild-scheduled"

RebuildScheduleAction = Literal["create", "update", "pause", "noop"]


class RebuildScheduleSpec(BaseModel, frozen=True):
    schedule_id: str
    action_id: str
    enabled: bool
    cron: str
    timezone: str


class ExistingRebuildSchedule(BaseModel, frozen=True):
    exists: bool
    cron: str | None = None
    timezone: str | None = None
    paused: bool | None = None


def derive_rebuild_schedule_spec(*, enabled: bool, cron: str, timezone: str) -> RebuildScheduleSpec:
    if not cron.strip():
        raise ValueError("Search index rebuild schedule cron must not be empty")
    if not timezone.strip():
        raise ValueError("Search index rebuild schedule timezone must not be empty")
    return RebuildScheduleSpec(
        schedule_id=REBUILD_SCHEDULE_ID,
        action_id=REBUILD_ACTION_ID,
        enabled=enabled,
        cron=cron,
        timezone=timezone,
    )


def plan_rebuild_schedule(
    *, spec: RebuildScheduleSpec, existing: ExistingRebuildSchedule
) -> RebuildScheduleAction:
    if spec.enabled:
        if not existing.exists:
            return "create"
        if existing.cron != spec.cron or existing.timezone != spec.timezone or existing.paused:
            return "update"
        return "noop"

    if existing.exists and not existing.paused:
        return "pause"
    return "noop"
