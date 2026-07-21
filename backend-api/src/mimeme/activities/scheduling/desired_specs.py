from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import IngestionSource
from mimeme.domain.source_schedule_spec import ScheduleSpec, derive_schedule_spec


def load_desired_specs(db: Session) -> list[ScheduleSpec]:

    sources = db.scalars(select(IngestionSource).where(IngestionSource.deleted_at.is_(None))).all()

    return [
        derive_schedule_spec(
            source_id=source.id,
            schedule_cron=source.schedule_cron,
            schedule_timezone=source.schedule_timezone,
            enabled=source.enabled,
            deleted=False,
        )
        for source in sources
    ]
