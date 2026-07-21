from __future__ import annotations

import datetime

from sqlalchemy.orm import Session

from mimeme.activities.scheduling.desired_specs import load_desired_specs
from mimeme.domain.source_schedule_spec import DesiredScheduleState
from tests.factories import create_ingestion_source


def _by_id(specs):
    return {spec.schedule_id: spec for spec in specs}


def test_load_desired_specs_maps_each_non_deleted_source(db_session: Session) -> None:
    enabled = create_ingestion_source(
        session=db_session, name="enabled", enabled=True, schedule_cron="0 * * * *"
    )
    disabled = create_ingestion_source(
        session=db_session, name="disabled", enabled=False, schedule_cron="0 * * * *"
    )
    no_cron = create_ingestion_source(
        session=db_session, name="no-cron", enabled=True, schedule_cron=None
    )
    deleted = create_ingestion_source(
        session=db_session,
        name="deleted",
        enabled=True,
        deleted_at=datetime.datetime.now(datetime.UTC),
    )

    specs = _by_id(load_desired_specs(db_session))

    assert specs[f"source-sync-{enabled.id}"].desired_state == DesiredScheduleState.ACTIVE
    assert specs[f"source-sync-{disabled.id}"].desired_state == DesiredScheduleState.PAUSED
    assert specs[f"source-sync-{no_cron.id}"].desired_state == DesiredScheduleState.ABSENT
    assert f"source-sync-{deleted.id}" not in specs
