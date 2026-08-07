from types import SimpleNamespace

from mimeme.index import rule
from mimeme.index.schedule import Desired, Existing, Spec, Temporal, changed


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
    absent = spec.model_copy(update={"cron": None})
    assert absent.desired is Desired.ABSENT
    assert changed(absent, Existing(exists=True))
    assert not changed(absent, Existing(exists=False))


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


async def test_temporal_schedule_deletes_an_existing_absent_schedule() -> None:
    class Handle:
        deleted = False

        async def describe(self):  # noqa: ANN202
            return SimpleNamespace(
                schedule=SimpleNamespace(
                    spec=SimpleNamespace(cron_expressions=["0 2 * * *"], time_zone_name="UTC"),
                    state=SimpleNamespace(paused=False),
                )
            )

        async def delete(self) -> None:
            self.deleted = True

    class Client:
        handle = Handle()

        def get_schedule_handle(self, schedule_id: str) -> Handle:
            assert schedule_id == rule.SCHEDULE_ID
            return self.handle

    client = Client()
    changed_schedule = await Temporal(client).reconcile(  # type: ignore[arg-type]
        Spec(enabled=False, cron=None, timezone="UTC"),
        model="test/embed",
        index_type="flat",
    )

    assert changed_schedule is True
    assert client.handle.deleted is True
