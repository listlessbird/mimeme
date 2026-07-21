from __future__ import annotations

from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

from mimeme.db.schema import ProcessingStatus
from mimeme.domain.source_item_browse import SourceItemBrowser
from mimeme.shared.services.media_url import MediaUrlResolver

pytestmark = pytest.mark.usefixtures(
    "_patch_domain_session_scope", "_patch_async_domain_session_scope"
)

MEDIA_URLS = MediaUrlResolver("https://assets.mimeme.dev")


class FakeApiStorage:
    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return True


async def test_list_items_empty_for_source(run_sync_seed, mock_storage: MagicMock) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)

    page = await SourceItemBrowser(MEDIA_URLS).list_items(source_id, limit=20, offset=0)

    assert page.items == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0


async def test_list_items_uses_public_media_url(run_sync_seed) -> None:
    def seed(session) -> int:
        source = create_ingestion_source(session=session)
        item = create_source_item(session=session, source=source, source_id=source.id)
        image = create_image(session=session, s3_key="images/test/source-item.jpg")
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            source_id=source.id,
            source_item_id=item.id,
            image_id=image.id,
            status=ProcessingStatus.DONE,
        )
        return source.id

    source_id = await run_sync_seed(seed)

    page = await SourceItemBrowser(MEDIA_URLS).list_items(source_id, limit=20, offset=0)

    assert page.items[0].thumbnail_url == ("https://assets.mimeme.dev/images/test/source-item.jpg")


async def test_list_run_items_uses_public_media_url(run_sync_seed) -> None:
    def seed(session) -> tuple[int, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source, source_id=source.id)
        item = create_source_item(session=session, source=source, source_id=source.id)
        image = create_image(session=session, s3_key="images/test/run-item.jpg")
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            image_id=image.id,
            status=ProcessingStatus.DONE,
        )
        return source.id, run.id

    source_id, run_id = await run_sync_seed(seed)

    page = await SourceItemBrowser(MEDIA_URLS).list_run_items(source_id, run_id, limit=20, offset=0)

    assert page.items[0].thumbnail_url == ("https://assets.mimeme.dev/images/test/run-item.jpg")
