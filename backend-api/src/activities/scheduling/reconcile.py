from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from activities.scheduling.desired_specs import load_desired_specs
from domain.source_schedule_reconcile import (
    CreateSchedule,
    DeleteSchedule,
    ExistingSchedule,
    UpdateSchedule,
    plan_reconciliation,
)
from domain.source_schedule_spec import ScheduleSpec


class ScheduleStore(Protocol):
    async def list_existing(self) -> list[ExistingSchedule]: ...
    async def create(self, spec: ScheduleSpec) -> None: ...
    async def update(self, spec: ScheduleSpec) -> None: ...
    async def delete(self, schedule_id: str) -> None: ...


async def reconcile(store: ScheduleStore, *, desired: list[ScheduleSpec]) -> None:

    existing = await store.list_existing()
    plan = plan_reconciliation(desired=desired, existing=existing)

    for action in plan:
        match action:
            case CreateSchedule(spec=schedule_spec):
                await store.create(schedule_spec)

            case UpdateSchedule(spec=schedule_spec):
                await store.update(schedule_spec)

            case DeleteSchedule(schedule_id=schedule_id):
                await store.delete(schedule_id)


async def run_reconciliation(db: Session, store: ScheduleStore) -> None:
    desired = load_desired_specs(db)
    await reconcile(store, desired=desired)
