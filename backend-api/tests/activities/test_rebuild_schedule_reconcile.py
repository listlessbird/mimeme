from __future__ import annotations

from mimeme.activities.scheduling.rebuild_schedule import reconcile_rebuild_schedule
from mimeme.domain.index_rebuild_schedule import (
    ExistingRebuildSchedule,
    RebuildScheduleSpec,
    derive_rebuild_schedule_spec,
)


class FakeStore:
    def __init__(self, existing: ExistingRebuildSchedule) -> None:
        self._existing = existing
        self.calls: list[str] = []

    async def describe(self) -> ExistingRebuildSchedule:
        return self._existing

    async def create(self, spec: RebuildScheduleSpec) -> None:
        self.calls.append("create")

    async def update(self, spec: RebuildScheduleSpec) -> None:
        self.calls.append("update")

    async def pause(self, schedule_id: str) -> None:
        self.calls.append("pause")


def _spec(enabled: bool = True, cron: str = "* * * * *"):
    return derive_rebuild_schedule_spec(enabled=enabled, cron=cron, timezone="UTC")


async def test_creates_when_absent() -> None:
    store = FakeStore(ExistingRebuildSchedule(exists=False))

    action = await reconcile_rebuild_schedule(store, spec=_spec())

    assert action == "create"
    assert store.calls == ["create"]


async def test_updates_on_cron_change() -> None:
    store = FakeStore(
        ExistingRebuildSchedule(exists=True, cron="*/5 * * * *", timezone="UTC", paused=False)
    )

    action = await reconcile_rebuild_schedule(store, spec=_spec())

    assert action == "update"
    assert store.calls == ["update"]


async def test_pauses_when_disabled() -> None:
    store = FakeStore(
        ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=False)
    )

    action = await reconcile_rebuild_schedule(store, spec=_spec(enabled=False))

    assert action == "pause"
    assert store.calls == ["pause"]


async def test_rerun_makes_no_unnecessary_change() -> None:
    store = FakeStore(
        ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=False)
    )

    action = await reconcile_rebuild_schedule(store, spec=_spec())

    assert action == "noop"
    assert store.calls == []
