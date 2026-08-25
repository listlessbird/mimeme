from __future__ import annotations

from temporalio.client import ScheduleActionStartWorkflow

from mimeme.source import rule
from mimeme.source.schedule import (
    CreateSchedule,
    DeleteSchedule,
    DesiredScheduleState,
    ExistingSchedule,
    ScheduleSpec,
    TemporalScheduleStore,
    UpdateSchedule,
    derive_schedule_spec,
    plan_reconciliation,
    reconcile,
)


def _spec(source_id: int, *, cron="0 * * * *", tz="UTC", state=DesiredScheduleState.ACTIVE):
    return derive_schedule_spec(
        source_id=source_id,
        schedule_cron=cron,
        schedule_timezone=tz,
        enabled=state != DesiredScheduleState.PAUSED,
        deleted=state == DesiredScheduleState.ABSENT,
    )


class TestDeriveSpec:
    def test_active_when_enabled_with_cron(self) -> None:
        spec = derive_schedule_spec(
            source_id=1,
            schedule_cron="0 * * * *",
            schedule_timezone="UTC",
            enabled=True,
            deleted=False,
        )
        assert spec.desired_state == DesiredScheduleState.ACTIVE
        assert spec.schedule_id == rule.schedule_id(1)

    def test_paused_when_disabled(self) -> None:
        spec = derive_schedule_spec(
            source_id=1,
            schedule_cron="0 * * * *",
            schedule_timezone="UTC",
            enabled=False,
            deleted=False,
        )
        assert spec.desired_state == DesiredScheduleState.PAUSED

    def test_absent_without_cron(self) -> None:
        spec = derive_schedule_spec(
            source_id=1, schedule_cron=None, schedule_timezone="UTC", enabled=True, deleted=False
        )
        assert spec.desired_state == DesiredScheduleState.ABSENT


def test_temporal_schedule_uses_configured_task_queue() -> None:
    store = TemporalScheduleStore(object(), task_queue="mimeme-v2")  # type: ignore[arg-type]
    temporal_schedule = store._build_schedule(_spec(1))  # noqa: SLF001

    assert isinstance(temporal_schedule.action, ScheduleActionStartWorkflow)
    assert temporal_schedule.action.task_queue == "mimeme-v2"


class TestPlanReconciliation:
    def test_create_missing(self) -> None:
        actions = plan_reconciliation(desired=[_spec(1)], existing=[])
        assert isinstance(actions[0], CreateSchedule)

    def test_noop_when_matching(self) -> None:
        spec = _spec(1)
        existing = ExistingSchedule(
            schedule_id=spec.schedule_id, cron=spec.cron, timezone="UTC", paused=False
        )
        assert plan_reconciliation(desired=[spec], existing=[existing]) == []

    def test_timezone_drift_triggers_update(self) -> None:
        spec = _spec(1, tz="America/New_York")
        existing = ExistingSchedule(
            schedule_id=spec.schedule_id, cron=spec.cron, timezone="UTC", paused=False
        )
        actions = plan_reconciliation(desired=[spec], existing=[existing])
        assert len(actions) == 1 and isinstance(actions[0], UpdateSchedule)

    def test_cron_drift_triggers_update(self) -> None:
        spec = _spec(1, cron="*/5 * * * *")
        existing = ExistingSchedule(
            schedule_id=spec.schedule_id, cron="0 * * * *", timezone="UTC", paused=False
        )
        actions = plan_reconciliation(desired=[spec], existing=[existing])
        assert isinstance(actions[0], UpdateSchedule)

    def test_pause_drift_triggers_update(self) -> None:
        spec = _spec(1, state=DesiredScheduleState.PAUSED)
        existing = ExistingSchedule(
            schedule_id=spec.schedule_id, cron=spec.cron, timezone="UTC", paused=False
        )
        actions = plan_reconciliation(desired=[spec], existing=[existing])
        assert isinstance(actions[0], UpdateSchedule)

    def test_absent_desired_deletes_existing(self) -> None:
        spec = _spec(1, state=DesiredScheduleState.ABSENT)
        existing = ExistingSchedule(
            schedule_id=spec.schedule_id, cron="0 * * * *", timezone="UTC", paused=False
        )
        actions = plan_reconciliation(desired=[spec], existing=[existing])
        assert isinstance(actions[0], DeleteSchedule)

    def test_orphan_existing_is_deleted(self) -> None:
        existing = ExistingSchedule(
            schedule_id=rule.schedule_id(99), cron="0 * * * *", timezone="UTC", paused=False
        )
        actions = plan_reconciliation(desired=[], existing=[existing])
        assert actions == [DeleteSchedule(schedule_id=existing.schedule_id)]


class _FakeStore:
    def __init__(self, existing: list[ExistingSchedule]) -> None:
        self._existing = existing
        self.created: list[ScheduleSpec] = []
        self.updated: list[ScheduleSpec] = []
        self.deleted: list[str] = []

    async def list_existing(self) -> list[ExistingSchedule]:
        return self._existing

    async def create(self, spec: ScheduleSpec) -> None:
        self.created.append(spec)

    async def update(self, spec: ScheduleSpec) -> None:
        self.updated.append(spec)

    async def delete(self, schedule_id: str) -> None:
        self.deleted.append(schedule_id)


class TestReconcileDriver:
    async def test_applies_planned_actions(self) -> None:
        store = _FakeStore(
            existing=[
                ExistingSchedule(
                    schedule_id=rule.schedule_id(2),
                    cron="0 * * * *",
                    timezone="UTC",
                    paused=False,
                ),
                ExistingSchedule(
                    schedule_id=rule.schedule_id(9),
                    cron="0 * * * *",
                    timezone="UTC",
                    paused=False,
                ),
            ]
        )
        await reconcile(store, desired=[_spec(1), _spec(2, tz="Europe/Berlin")])
        assert [s.source_id for s in store.created] == [1]
        assert [s.source_id for s in store.updated] == [2]
        assert store.deleted == [rule.schedule_id(9)]
