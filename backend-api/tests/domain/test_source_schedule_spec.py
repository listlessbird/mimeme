from __future__ import annotations

import pytest

from domain.source_schedule_spec import (
    DesiredScheduleState,
    ScheduleOverlapPolicy,
    ScheduleSpec,
    derive_schedule_spec,
)


def test_enabled_source_with_cron_is_active_with_deterministic_id() -> None:
    # The tracer bullet: a healthy Source maps to a fully-specified ACTIVE
    # schedule. Deterministic id, the source's cron/timezone, SKIP overlap so a
    # slow run never piles up behind the next fire.
    spec = derive_schedule_spec(
        source_id=1,
        schedule_cron="0 * * * *",
        schedule_timezone="UTC",
        enabled=True,
        deleted=False,
    )

    assert spec == ScheduleSpec(
        source_id=1,
        schedule_id="source-sync-1",
        cron="0 * * * *",
        timezone="UTC",
        overlap_policy=ScheduleOverlapPolicy.SKIP,
        desired_state=DesiredScheduleState.ACTIVE,
    )


def test_spec_carries_source_id_for_the_workflow_action() -> None:
    # The apply layer starts SourceSyncWorkflow(source_id=...). The spec must
    # carry source_id directly, not force the adapter to parse it back out of the
    # "source-sync-{id}" schedule_id string.
    spec = derive_schedule_spec(
        source_id=42,
        schedule_cron="0 * * * *",
        schedule_timezone="UTC",
        enabled=True,
        deleted=False,
    )

    assert spec.source_id == 42
    assert spec.schedule_id == "source-sync-42"


def test_disabled_source_is_paused_but_keeps_its_cron() -> None:
    # Disabling pauses the Schedule without tearing it down: the cron is retained
    # so re-enabling resumes the same Schedule rather than rebuilding it.
    spec = derive_schedule_spec(
        source_id=7,
        schedule_cron="*/15 * * * *",
        schedule_timezone="UTC",
        enabled=False,
        deleted=False,
    )

    assert spec.desired_state == DesiredScheduleState.PAUSED
    assert spec.cron == "*/15 * * * *"
    assert spec.schedule_id == "source-sync-7"


def test_deleted_source_is_absent_even_when_still_enabled() -> None:
    # Soft-delete tears the Schedule down, not pauses it. Deletion outranks the
    # enabled flag: a deleted Source is ABSENT even if its row still reads enabled.
    spec = derive_schedule_spec(
        source_id=3,
        schedule_cron="0 0 * * *",
        schedule_timezone="UTC",
        enabled=True,
        deleted=True,
    )

    assert spec.desired_state == DesiredScheduleState.ABSENT


def test_enabled_source_without_a_cron_is_absent() -> None:
    # A Source may exist with no cron: it runs only via manual trigger and has no
    # Temporal Schedule at all. Nothing to fire automatically -> ABSENT.
    spec = derive_schedule_spec(
        source_id=9,
        schedule_cron=None,
        schedule_timezone="UTC",
        enabled=True,
        deleted=False,
    )

    assert spec.desired_state == DesiredScheduleState.ABSENT
    assert spec.cron is None


@pytest.mark.parametrize(
    ("enabled", "deleted", "has_cron", "expected"),
    [
        (True, False, True, DesiredScheduleState.ACTIVE),
        (False, False, True, DesiredScheduleState.PAUSED),
        (True, True, True, DesiredScheduleState.ABSENT),  # deleted outranks enabled
        (False, True, True, DesiredScheduleState.ABSENT),  # deleted outranks disabled
        (True, False, False, DesiredScheduleState.ABSENT),  # no cron -> nothing to fire
        (False, True, False, DesiredScheduleState.ABSENT),
    ],
)
def test_desired_state_decision_table(
    enabled: bool, deleted: bool, has_cron: bool, expected: DesiredScheduleState
) -> None:
    spec = derive_schedule_spec(
        source_id=1,
        schedule_cron="0 * * * *" if has_cron else None,
        schedule_timezone="UTC",
        enabled=enabled,
        deleted=deleted,
    )

    assert spec.desired_state == expected


def test_spec_is_frozen() -> None:
    spec = derive_schedule_spec(
        source_id=1,
        schedule_cron="0 * * * *",
        schedule_timezone="UTC",
        enabled=True,
        deleted=False,
    )

    with pytest.raises((AttributeError, TypeError, ValueError)):
        spec.desired_state = DesiredScheduleState.ABSENT  # type: ignore[misc]
