from mimeme.index.schedule import Existing, Spec, changed


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
