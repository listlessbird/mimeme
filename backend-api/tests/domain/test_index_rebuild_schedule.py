from __future__ import annotations

import pytest

from mimeme.domain.index_rebuild_schedule import (
    REBUILD_ACTION_ID,
    REBUILD_SCHEDULE_ID,
    ExistingRebuildSchedule,
    derive_rebuild_schedule_spec,
    plan_rebuild_schedule,
)


def _spec(enabled: bool = True, cron: str = "* * * * *", timezone: str = "UTC"):
    return derive_rebuild_schedule_spec(enabled=enabled, cron=cron, timezone=timezone)


def test_derive_uses_fixed_identity() -> None:
    spec = _spec()
    assert spec.schedule_id == REBUILD_SCHEDULE_ID == "search-index-rebuild"
    assert spec.action_id == REBUILD_ACTION_ID == "search-index-rebuild-scheduled"


def test_derive_rejects_empty_cron() -> None:
    with pytest.raises(ValueError):
        derive_rebuild_schedule_spec(enabled=True, cron="  ", timezone="UTC")


def test_derive_rejects_empty_timezone() -> None:
    with pytest.raises(ValueError):
        derive_rebuild_schedule_spec(enabled=True, cron="* * * * *", timezone="")


def test_enabled_and_absent_creates() -> None:
    action = plan_rebuild_schedule(spec=_spec(), existing=ExistingRebuildSchedule(exists=False))
    assert action == "create"


def test_enabled_and_matching_is_noop() -> None:
    existing = ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=False)
    assert plan_rebuild_schedule(spec=_spec(), existing=existing) == "noop"


def test_enabled_with_changed_cron_updates() -> None:
    existing = ExistingRebuildSchedule(
        exists=True, cron="*/5 * * * *", timezone="UTC", paused=False
    )
    assert plan_rebuild_schedule(spec=_spec(), existing=existing) == "update"


def test_enabled_with_changed_timezone_updates() -> None:
    existing = ExistingRebuildSchedule(
        exists=True, cron="* * * * *", timezone="America/New_York", paused=False
    )
    assert plan_rebuild_schedule(spec=_spec(), existing=existing) == "update"


def test_enabled_but_paused_updates_to_unpause() -> None:
    existing = ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=True)
    assert plan_rebuild_schedule(spec=_spec(), existing=existing) == "update"


def test_disabled_and_running_pauses() -> None:
    existing = ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=False)
    assert plan_rebuild_schedule(spec=_spec(enabled=False), existing=existing) == "pause"


def test_disabled_and_already_paused_is_noop() -> None:
    existing = ExistingRebuildSchedule(exists=True, cron="* * * * *", timezone="UTC", paused=True)
    assert plan_rebuild_schedule(spec=_spec(enabled=False), existing=existing) == "noop"


def test_disabled_and_absent_is_noop() -> None:
    action = plan_rebuild_schedule(
        spec=_spec(enabled=False), existing=ExistingRebuildSchedule(exists=False)
    )
    assert action == "noop"
