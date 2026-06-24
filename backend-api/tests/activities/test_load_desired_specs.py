from __future__ import annotations

import datetime

from sqlalchemy.orm import Session
from tests.factories import create_ingestion_source

from activities.scheduling.desired_specs import load_desired_specs
from domain.source_schedule_spec import DesiredScheduleState


def _by_id(specs):
    return {spec.schedule_id: spec for spec in specs}


def test_load_desired_specs_maps_each_non_deleted_source(db_session: Session) -> None:
    # The bridge from Postgres to the domain: every live Source becomes its
    # desired spec (active/paused/absent), keyed by deterministic id. A
    # soft-deleted Source is excluded entirely -- its Schedule is left to the
    # planner's orphan rule, not represented as a desired spec.
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
