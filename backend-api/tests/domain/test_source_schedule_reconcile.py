from __future__ import annotations

from mimeme.domain.source_schedule_reconcile import (
    CreateSchedule,
    DeleteSchedule,
    ExistingSchedule,
    UpdateSchedule,
    plan_reconciliation,
)
from mimeme.domain.source_schedule_spec import derive_schedule_spec


def _spec(
    source_id: int, *, enabled: bool = True, deleted: bool = False, cron: str | None = "0 * * * *"
):
    return derive_schedule_spec(
        source_id=source_id,
        schedule_cron=cron,
        schedule_timezone="UTC",
        enabled=enabled,
        deleted=deleted,
    )


def test_enabled_source_with_no_schedule_yet_is_created() -> None:
    # Reconciliation's tracer bullet: a Source that should be running but whose
    # Temporal Schedule is missing (e.g. an inline write that never landed) is
    # restored by a Create.
    spec = _spec(1)

    plan = plan_reconciliation(desired=[spec], existing=[])

    assert plan == [CreateSchedule(spec=spec)]


def test_already_converged_active_schedule_yields_no_action() -> None:
    # Idempotency: the Schedule already exists with the right cron and is not
    # paused, so a repeated sweep must do nothing -- no churn.
    spec = _spec(1)
    existing = ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=False)

    plan = plan_reconciliation(desired=[spec], existing=[existing])

    assert plan == []


def test_schedule_with_drifted_cron_is_updated() -> None:
    # The "update stale" half of reconciliation: the Schedule exists but its cron
    # no longer matches the Source's config, so it is converged via an Update
    # carrying the desired spec.
    spec = _spec(1, cron="*/30 * * * *")
    stale = ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=False)

    plan = plan_reconciliation(desired=[spec], existing=[stale])

    assert plan == [UpdateSchedule(spec=spec)]


def test_active_source_whose_schedule_is_paused_is_updated_to_resume() -> None:
    # Re-enabling: the Schedule is present and correctly cronned but still paused
    # from an earlier disable. Reconciliation resumes it via an Update.
    spec = _spec(1)
    paused = ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=True)

    plan = plan_reconciliation(desired=[spec], existing=[paused])

    assert plan == [UpdateSchedule(spec=spec)]


def test_absent_spec_with_existing_schedule_is_deleted() -> None:
    # A deleted (or no-cron) Source yields an ABSENT spec; if a Schedule still
    # lingers for it, tear it down.
    spec = _spec(1, deleted=True)
    lingering = ExistingSchedule(schedule_id="source-sync-1", cron="0 * * * *", paused=False)

    plan = plan_reconciliation(desired=[spec], existing=[lingering])

    assert plan == [DeleteSchedule(schedule_id="source-sync-1")]


def test_absent_spec_with_no_schedule_does_nothing() -> None:
    # Nothing to tear down: don't emit a delete for a Schedule that isn't there.
    spec = _spec(1, deleted=True)

    plan = plan_reconciliation(desired=[spec], existing=[])

    assert plan == []


def test_orphan_schedule_with_no_matching_spec_is_deleted() -> None:
    # An orphan: Temporal has a Schedule whose Source is absent from the desired
    # set entirely (e.g. hard-gone). Reconciliation removes it.
    orphan = ExistingSchedule(schedule_id="source-sync-99", cron="0 * * * *", paused=False)

    plan = plan_reconciliation(desired=[], existing=[orphan])

    assert plan == [DeleteSchedule(schedule_id="source-sync-99")]
