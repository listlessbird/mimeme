from mimeme.index import rule
from mimeme.index.schedule import Existing, Spec, Temporal, changed


def test_schedule_reconciliation_covers_timezone_pause_and_noop() -> None:
    spec = Spec(enabled=True, cron="0 2 * * *", timezone="Asia/Kolkata")

    assert changed(spec, Existing(exists=False))
    assert changed(
        spec,
        Existing(exists=True, cron=spec.cron, timezone="UTC", paused=False),
    )
    assert changed(
        spec.model_copy(update={"enabled": False}),
        Existing(exists=True, cron=spec.cron, timezone=spec.timezone, paused=False),
    )
    assert not changed(
        spec,
        Existing(exists=True, cron=spec.cron, timezone=spec.timezone, paused=False),
    )


def test_temporal_schedule_uses_exact_ids_queue_and_skip_overlap() -> None:
    adapter = Temporal(object())  # type: ignore[arg-type]
    schedule = adapter._schedule(  # noqa: SLF001
        Spec(enabled=True, cron="0 2 * * *", timezone="Asia/Kolkata"),
        model="test/embed",
        index_type="flat",
    )

    assert schedule.action.id == rule.workflow_id("scheduled")
    assert schedule.action.task_queue == "mimeme-v2"
    assert schedule.policy.overlap.name == "SKIP"
