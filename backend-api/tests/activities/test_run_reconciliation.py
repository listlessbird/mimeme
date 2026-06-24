from __future__ import annotations

from sqlalchemy.orm import Session
from tests.activities.test_schedule_reconcile import FakeScheduleStore, _by_id
from tests.factories import create_ingestion_source

from activities.scheduling.reconcile import run_reconciliation
from domain.source_schedule_reconcile import ExistingSchedule


async def test_run_reconciliation_converges_temporal_to_the_live_sources(
    db_session: Session,
) -> None:
    # End to end through both real seams (Postgres + the ScheduleStore): the live
    # Sources in the DB drive the desired set, and the store is converged to match
    # -- missing created, paused fixed, orphan removed.
    enabled = create_ingestion_source(
        session=db_session, name="enabled", enabled=True, schedule_cron="0 * * * *"
    )
    disabled = create_ingestion_source(
        session=db_session, name="disabled", enabled=False, schedule_cron="0 * * * *"
    )
    create_ingestion_source(session=db_session, name="kept-history", deleted_at=_now())

    store = FakeScheduleStore(
        initial=[
            # disabled Source's Schedule is still running -> must be paused
            ExistingSchedule(
                schedule_id=f"source-sync-{disabled.id}", cron="0 * * * *", paused=False
            ),
            # a Schedule whose Source no longer exists -> orphan, delete
            ExistingSchedule(schedule_id="source-sync-9999", cron="0 * * * *", paused=False),
        ]
    )

    await run_reconciliation(db_session, store)

    assert _by_id(await store.list_existing()) == {
        f"source-sync-{enabled.id}": ExistingSchedule(
            schedule_id=f"source-sync-{enabled.id}", cron="0 * * * *", paused=False
        ),
        f"source-sync-{disabled.id}": ExistingSchedule(
            schedule_id=f"source-sync-{disabled.id}", cron="0 * * * *", paused=True
        ),
    }


def _now():
    import datetime

    return datetime.datetime.now(datetime.UTC)
