"""Schema contract for pHash dedup provenance on ingest URLs.

Two structural changes are pinned here:

1. The `images -> ingest_urls` relationship is now **one-to-many**. A dedup
   legitimately maps many ingest URLs onto one canonical image, so an Image
   must expose a *collection* of ingest URLs, not a single one.

2. An IngestURL can record *why* it was treated as a duplicate
   (`duplicate_reason`) and *which* canonical image it was collapsed into
   (`duplicate_of_image_id`). Manual uploads leave both null.

These run against the schema `conftest` builds from the ORM metadata, so they
verify the ORM mapping (relationship + columns), not the Alembic migration
itself — the migration is verified by running it.
"""

from __future__ import annotations

from sqlalchemy.orm import Session
from tests.factories import create_image, create_ingest_url, create_job

from mimeme.db.schema import DuplicateReason


def test_many_ingest_urls_can_point_at_one_canonical_image(db_session: Session) -> None:
    """A canonical image can be referenced by more than one ingest URL."""
    image = create_image(session=db_session)
    job = create_job(session=db_session)

    url_a = create_ingest_url(session=db_session, job=job, image=image)
    url_b = create_ingest_url(session=db_session, job=job, image=image)
    db_session.flush()
    db_session.refresh(image)

    assert {u.id for u in image.ingest_urls} == {url_a.id, url_b.id}


def test_ingest_url_records_the_duplicate_decision(db_session: Session) -> None:
    """When deduped, an ingest URL stores the reason and the canonical image."""
    canonical = create_image(session=db_session)
    job = create_job(session=db_session)

    url = create_ingest_url(
        session=db_session,
        job=job,
        image=canonical,
        duplicate_reason=DuplicateReason.PHASH,
        duplicate_of_image_id=canonical.id,
    )
    db_session.flush()
    db_session.refresh(url)

    assert url.duplicate_reason == DuplicateReason.PHASH
    assert url.duplicate_of_image_id == canonical.id


def test_manual_upload_leaves_provenance_null(db_session: Session) -> None:
    """A plain ingest URL (manual upload) has no duplicate provenance."""
    job = create_job(session=db_session)

    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    assert url.duplicate_reason is None
    assert url.duplicate_of_image_id is None
