"""Schema spec for the Source registry tables (issue 02).

Pins the ORM mapping for `ingestion_sources`, `source_runs`, `source_items`,
and the new `ingest_urls` provenance FKs. These run against the schema
`conftest` builds from the ORM metadata, so they verify the mapping
(columns, relationships, constraints) — not the Alembic migration, which is
verified by running it.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.models import (
    IngestionSource,
    SourceItem,
    SourceRunStatus,
    SourceRunTrigger,
    SourceType,
)
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


def test_source_defaults_to_api_type_enabled_and_undeleted(db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    db_session.flush()
    db_session.refresh(source)

    assert source.source_type == SourceType.API
    assert source.enabled is True
    assert source.deleted_at is None
    assert source.created_at is not None


def test_two_live_sources_cannot_share_a_name(db_session: Session) -> None:
    create_ingestion_source(session=db_session, name="dupe")
    db_session.flush()

    # Built directly (not via factory) so the conflicting INSERT fires inside
    # the pytest.raises block rather than during the factory's auto-flush.
    db_session.add(IngestionSource(name="dupe", adapter_key="meme_api"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_soft_deleted_name_is_freed_for_reuse(db_session: Session) -> None:
    # The point of the partial unique index: deleting a Source releases its name.
    source = create_ingestion_source(session=db_session, name="reuse")
    db_session.flush()
    source.deleted_at = datetime.datetime.now(datetime.UTC)
    db_session.flush()

    create_ingestion_source(session=db_session, name="reuse")
    db_session.flush()  # no IntegrityError: the tombstone no longer holds the name


def test_source_owns_its_runs_and_items(db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    run = create_source_run(session=db_session, source=source)
    item = create_source_item(session=db_session, source=source)
    db_session.flush()
    db_session.refresh(source)

    assert [r.id for r in source.runs] == [run.id]
    assert [i.id for i in source.items] == [item.id]


def test_source_run_defaults_to_pending(db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    run = create_source_run(
        session=db_session, source=source, trigger_mode=SourceRunTrigger.SCHEDULED
    )
    db_session.flush()
    db_session.refresh(run)

    assert run.status == SourceRunStatus.PENDING
    assert run.trigger_mode == SourceRunTrigger.SCHEDULED
    assert run.ingest_job_id is None


def test_source_item_unique_per_source_and_external_id(db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    create_source_item(session=db_session, source=source, external_item_id="abc")
    db_session.flush()

    # Built directly (not via factory) so the conflicting INSERT fires inside
    # the pytest.raises block rather than during the factory's auto-flush.
    db_session.add(SourceItem(source_id=source.id, external_item_id="abc"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_same_external_id_allowed_across_sources(db_session: Session) -> None:
    source_a = create_ingestion_source(session=db_session)
    source_b = create_ingestion_source(session=db_session)
    create_source_item(session=db_session, source=source_a, external_item_id="shared")
    create_source_item(session=db_session, source=source_b, external_item_id="shared")
    db_session.flush()  # no IntegrityError: uniqueness is per (source, external_id)


def test_ingest_url_carries_source_provenance(db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    run = create_source_run(session=db_session, source=source)
    item = create_source_item(session=db_session, source=source)
    job = create_job(session=db_session)
    db_session.flush()

    url = create_ingest_url(
        session=db_session,
        job=job,
        source_id=source.id,
        source_run_id=run.id,
        source_item_id=item.id,
    )
    db_session.flush()
    db_session.refresh(url)

    assert url.source_id == source.id
    assert url.source_run_id == run.id
    assert url.source_item_id == item.id


def test_manual_upload_leaves_source_provenance_null(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    assert url.source_id is None
    assert url.source_run_id is None
    assert url.source_item_id is None


def test_image_provenance_does_not_break_image_ingest_urls(db_session: Session) -> None:
    """The new ingest_urls FKs must not disturb the existing image->urls link."""
    image = create_image(session=db_session)
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job, image=image)
    db_session.flush()
    db_session.refresh(image)

    assert [u.id for u in image.ingest_urls] == [url.id]
