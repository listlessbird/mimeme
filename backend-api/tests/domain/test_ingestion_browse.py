from __future__ import annotations

from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from tests.factories import create_image, create_ingest_url, create_job

from domain.ingestion_browse import IngestionBrowser, IngestionView
from shared.config import settings
from shared.models.orm import ProcessingStatus

pytestmark = pytest.mark.usefixtures(
    "_patch_domain_session_scope", "_patch_async_domain_session_scope"
)


class FakeApiStorage:
    def __init__(self) -> None:
        self.presigned_keys: list[tuple[str, int]] = []

    def presign(self, key: str, expiration: int = 3600) -> str:
        self.presigned_keys.append((key, expiration))
        return f"https://fake/{key}"

    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return True


async def test_list_attempts_empty(mock_storage: MagicMock) -> None:
    page = await IngestionBrowser(mock_storage).list_attempts(
        limit=20,
        offset=0,
        view=IngestionView.ALL,
    )

    assert page.rows == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0


async def test_list_attempts_uses_api_storage_presign_surface(run_sync_seed) -> None:
    storage = FakeApiStorage()

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

    page = await IngestionBrowser(storage).list_attempts(
        limit=20,
        offset=0,
        view=IngestionView.ALL,
    )

    assert page.rows[0].thumbnail_url == "https://fake/images/test/example.jpg"
    assert storage.presigned_keys == [("images/test/example.jpg", settings.s3_presigned_url_expiry)]
