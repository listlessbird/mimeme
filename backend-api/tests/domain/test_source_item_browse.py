from __future__ import annotations

from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

from domain.source_item_browse import SourceItemBrowser
from shared.config import settings
from shared.models.orm import ProcessingStatus

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


class FakeApiStorage:
    def __init__(self) -> None:
        self.presigned_keys: list[tuple[str, int]] = []

    def presign(self, key: str, expiration: int = 3600) -> str:
        self.presigned_keys.append((key, expiration))
        return f"https://fake/{key}"

    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    def delete(self, key: str) -> None:
        pass

    def exists(self, key: str) -> bool:
        return True


def test_list_items_empty_for_source(db_session: Session, mock_storage: MagicMock) -> None:
    source = create_ingestion_source(session=db_session)
    db_session.flush()

    page = SourceItemBrowser(mock_storage).list_items(source.id, limit=20, offset=0)

    assert page.items == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0


def test_list_items_uses_api_storage_presign_surface(db_session: Session) -> None:
    storage = FakeApiStorage()
    source = create_ingestion_source(session=db_session)
    item = create_source_item(session=db_session, source=source, source_id=source.id)
    image = create_image(session=db_session, s3_key="images/test/source-item.jpg")
    job = create_job(session=db_session)
    create_ingest_url(
        session=db_session,
        job=job,
        job_id=job.id,
        source_id=source.id,
        source_item_id=item.id,
        image_id=image.id,
        status=ProcessingStatus.DONE,
    )
    db_session.flush()

    page = SourceItemBrowser(storage).list_items(source.id, limit=20, offset=0)

    assert page.items[0].thumbnail_url == "https://fake/images/test/source-item.jpg"
    assert storage.presigned_keys == [
        ("images/test/source-item.jpg", settings.s3_presigned_url_expiry)
    ]


def test_list_run_items_uses_api_storage_presign_surface(db_session: Session) -> None:
    storage = FakeApiStorage()
    source = create_ingestion_source(session=db_session)
    run = create_source_run(session=db_session, source=source, source_id=source.id)
    item = create_source_item(session=db_session, source=source, source_id=source.id)
    image = create_image(session=db_session, s3_key="images/test/run-item.jpg")
    job = create_job(session=db_session)
    create_ingest_url(
        session=db_session,
        job=job,
        job_id=job.id,
        source_id=source.id,
        source_run_id=run.id,
        source_item_id=item.id,
        image_id=image.id,
        status=ProcessingStatus.DONE,
    )
    db_session.flush()

    page = SourceItemBrowser(storage).list_run_items(source.id, run.id, limit=20, offset=0)

    assert page.items[0].thumbnail_url == "https://fake/images/test/run-item.jpg"
    assert storage.presigned_keys == [
        ("images/test/run-item.jpg", settings.s3_presigned_url_expiry)
    ]
