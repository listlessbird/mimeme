from __future__ import annotations

from activities.scheduling.reconcile import reconcile
from domain.source_schedule_reconcile import ExistingSchedule
from domain.source_schedule_spec import (
    DesiredScheduleState,
    ScheduleSpec,
    derive_schedule_spec,
)


class FakeScheduleStore:
    """In-memory ScheduleStore that models a Temporal Schedule set as a dict.

    A real seam, not a mock: tests assert the store's resulting state, never
    which methods were called.
    """

    def __init__(self, initial: list[ExistingSchedule] | None = None) -> None:
        self._by_id = {schedule.schedule_id: schedule for schedule in (initial or [])}

    async def list_existing(self) -> list[ExistingSchedule]:
        return list(self._by_id.values())

    async def create(self, spec: ScheduleSpec) -> None:
        self._by_id[spec.schedule_id] = self._to_existing(spec)

    async def update(self, spec: ScheduleSpec) -> None:
        self._by_id[spec.schedule_id] = self._to_existing(spec)

    async def delete(self, schedule_id: str) -> None:
        self._by_id.pop(schedule_id, None)

    @staticmethod
    def _to_existing(spec: ScheduleSpec) -> ExistingSchedule:
        return ExistingSchedule(
            schedule_id=spec.schedule_id,
            cron=spec.cron,
            paused=spec.desired_state == DesiredScheduleState.PAUSED,
        )


def _spec(
    source_id: int, *, enabled: bool = True, deleted: bool = False, cron: str | None = "0 * * * *"
) -> ScheduleSpec:
    return derive_schedule_spec(
        source_id=source_id,
        schedule_cron=cron,
        schedule_timezone="UTC",
        enabled=enabled,
        deleted=deleted,
    )


async def test_reconcile_creates_missing_schedule_for_enabled_source() -> None:
    # Tracer bullet for the imperative shell: an enabled Source with no Schedule
    # ends with one present in the store, unpaused, carrying its cron.
    store = FakeScheduleStore()

    await reconcile(store, desired=[_spec(1)])

    assert await store.list_existing() == [
        ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=False)
    ]


def _by_id(schedules: list[ExistingSchedule]) -> dict[str, ExistingSchedule]:
    return {schedule.schedule_id: schedule for schedule in schedules}


async def test_reconcile_converges_a_divergent_store_and_is_idempotent() -> None:
    # The authority sweep (criteria #5/#6): one store, every kind of drift at
    # once. After reconcile the store must match the desired set exactly, and a
    # second reconcile must be a no-op (idempotent).
    desired = [
        _spec(1),  # missing -> create
        _spec(2, cron="*/30 * * * *"),  # present but stale cron -> update
        _spec(3, enabled=False),  # present & active but should be paused -> update
        _spec(4),  # already converged -> no change
    ]
    store = FakeScheduleStore(
        initial=[
            ExistingSchedule(schedule_id="source-sync-2", cron="0 * * * *", paused=False),
            ExistingSchedule(schedule_id="source-sync-3", cron="0 * * * *", paused=False),
            ExistingSchedule(schedule_id="source-sync-4", cron="0 * * * *", paused=False),
            ExistingSchedule(schedule_id="source-sync-99", cron="0 * * * *", paused=False),  # orphan
        ]
    )

    await reconcile(store, desired=desired)

    converged = _by_id(await store.list_existing())
    assert converged == {
        "source-sync-1": ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=False),
        "source-sync-2": ExistingSchedule(schedule_id="source-sync-2", cron="*/30 * * * *", paused=False),
        "source-sync-3": ExistingSchedule(schedule_id="source-sync-3", cron="0 * * * *", paused=True),
        "source-sync-4": ExistingSchedule(schedule_id="source-sync-4", cron="0 * * * *", paused=False),
    }

    # Idempotency: the second pass plans nothing, so the state is unchanged.
    state_after_first = converged
    await reconcile(store, desired=desired)
    assert _by_id(await store.list_existing()) == state_after_first
