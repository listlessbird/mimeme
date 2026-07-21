from __future__ import annotations

from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from tests.factories import create_image, create_ingest_url, create_job

from mimeme.db.schema import ProcessingStatus
from mimeme.domain.ingestion_browse import IngestionBrowser, IngestionView
from mimeme.shared.services.media_url import MediaUrlResolver

MEDIA_URLS = MediaUrlResolver("https://assets.mimeme.dev")

pytestmark = pytest.mark.usefixtures(
    "_patch_domain_session_scope", "_patch_async_domain_session_scope"
)


class FakeApiStorage:
    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return True


async def test_list_attempts_empty(mock_storage: MagicMock) -> None:
    page = await IngestionBrowser(MEDIA_URLS).list_attempts(
        limit=20,
        offset=0,
        view=IngestionView.ALL,
    )

    assert page.rows == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0


async def test_list_attempts_uses_public_media_url(run_sync_seed) -> None:
    def seed(session) -> None:
        image = create_image(session=session, s3_key="images/test/example.jpg")
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            image_id=image.id,
            status=ProcessingStatus.DONE,
        )

    await run_sync_seed(seed)

    page = await IngestionBrowser(MEDIA_URLS).list_attempts(
        limit=20,
        offset=0,
        view=IngestionView.ALL,
    )

    assert page.rows[0].thumbnail_url == "https://assets.mimeme.dev/images/test/example.jpg"
