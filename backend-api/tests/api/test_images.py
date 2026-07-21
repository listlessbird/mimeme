"""Tests for the /images endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import IngestURL, Job, JobType, ProcessingStatus
from mimeme.shared.config import settings
from tests.factories import (
    create_annotation,
    create_image,
    create_processing,
)


class TestIngestImages:
    async def test_ingest_creates_job_and_urls(
        self,
        async_client: AsyncClient,
        async_db_session: AsyncSession,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post(
            "/images",
            json={
                "urls": ["https://example.com/img1.jpg", "https://example.com/img2.png"],
                "dataset": "test",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["queued"] == 2
        assert data["duplicates"] == 0
        assert "job_id" in data

        job = await async_db_session.get(Job, data["job_id"])
        assert job is not None
        assert job.type == JobType.INGEST

        urls = (
            await async_db_session.scalars(
                select(IngestURL).where(IngestURL.job_id == data["job_id"])
            )
        ).all()
        assert len(urls) == 2

    async def test_ingest_deduplicates_urls(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.post(
            "/images",
            json={
                "urls": [
                    "https://example.com/img1.jpg",
                    "https://example.com/img1.jpg",
                    "https://example.com/img2.png",
                ],
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["queued"] == 2
        assert data["duplicates"] == 1

    async def test_ingest_starts_temporal_workflow(
        self,
        async_client: AsyncClient,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post(
            "/images",
            json={"urls": ["https://example.com/img1.jpg"]},
        )
        assert resp.status_code == 202
        mock_temporal.start_workflow.assert_called_once()

    async def test_ingest_empty_urls_returns_422(self, async_client: AsyncClient) -> None:
        resp = await async_client.post("/images", json={"urls": []})
        assert resp.status_code == 422

    async def test_ingest_invalid_url_returns_422(self, async_client: AsyncClient) -> None:
        resp = await async_client.post("/images", json={"urls": ["not-a-url"]})
        assert resp.status_code == 422


class TestUploadImage:
    async def test_upload_stores_file_and_creates_ingest_job(
        self,
        async_client: AsyncClient,
        async_db_session: AsyncSession,
        api_storage,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post(
            "/images/upload",
            files={"file": ("meme.jpg", b"fake-image-bytes", "image/jpeg")},
            data={"dataset": "memes", "tags": ["funny", "cats"]},
        )

        assert resp.status_code == 202
        data = resp.json()
        assert data["queued"] == 1
        assert data["duplicates"] == 0

        assert len(api_storage.uploaded) == 1
        stored_key = api_storage.uploaded[0].key
        assert stored_key.startswith("uploads/staging/")

        urls = (
            await async_db_session.scalars(
                select(IngestURL).where(IngestURL.job_id == data["job_id"])
            )
        ).all()
        assert len(urls) == 1
        assert urls[0].input_kind == "staged_upload"
        assert urls[0].url is None
        assert urls[0].artifact_key == stored_key

    async def test_upload_starts_workflow(
        self,
        async_client: AsyncClient,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post(
            "/images/upload",
            files={"file": ("meme.png", b"bytes", "image/png")},
        )
        assert resp.status_code == 202
        mock_temporal.start_workflow.assert_called_once()

    async def test_upload_empty_file_returns_400(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/images/upload",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    async def test_upload_rejects_non_admin(
        self, async_client: AsyncClient, api_storage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "app_env", "production")
        monkeypatch.setattr(settings.http, "api_key_admin", SecretStr("secret-admin-key"))

        resp = await async_client.post(
            "/images/upload",
            files={"file": ("meme.jpg", b"bytes", "image/jpeg")},
        )

        assert resp.status_code == 403
        assert api_storage.uploaded == []


class TestListImages:
    async def test_list_images_empty(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.get("/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"] == []
        assert data["total"] == 0

    async def test_list_images_with_data(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_image(session=session),
                create_image(session=session),
            )
        )

        resp = await async_client.get("/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["images"]) == 2

    async def test_list_images_pagination(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(lambda session: [create_image(session=session) for _ in range(5)])

        resp = await async_client.get("/images?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 2
        assert data["total"] == 5
        assert data["has_more"] is True

    async def test_list_images_offset_beyond_total(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(lambda session: create_image(session=session))

        resp = await async_client.get("/images?offset=999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"] == []
        assert data["total"] == 1
        assert data["has_more"] is False

    async def test_list_images_filter_by_dataset(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_image(session=session, dataset="cats"),
                create_image(session=session, dataset="dogs"),
            )
        )

        resp = await async_client.get("/images?dataset=cats")
        data = resp.json()
        assert data["total"] == 1
        assert data["images"][0]["dataset"] == "cats"

    async def test_list_images_filter_by_status_done(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> None:
            img1 = create_image(session=session)
            create_processing(session=session, image=img1, embed_status=ProcessingStatus.DONE)
            img2 = create_image(session=session)
            create_processing(session=session, image=img2, embed_status=ProcessingStatus.PENDING)

        await run_sync_seed(seed)

        resp = await async_client.get("/images?status=done")
        data = resp.json()
        assert data["total"] == 1

    async def test_list_images_sort_oldest(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_image(session=session),
                create_image(session=session),
            )
        )

        resp = await async_client.get("/images?sort=oldest")
        data = resp.json()
        assert data["images"][0]["id"] <= data["images"][1]["id"]


class TestGetImage:
    async def test_get_existing_image(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> tuple[int, str]:
            image = create_image(session=session)
            return image.id, image.sha256

        image_id, sha256 = await run_sync_seed(seed)

        resp = await async_client.get(f"/images/{image_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == image_id
        assert data["sha256"] == sha256

    async def test_get_image_with_annotations(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> int:
            image = create_image(session=session)
            create_annotation(session=session, image=image, caption_text="A cat", ocr_text="LOL")
            return image.id

        image_id = await run_sync_seed(seed)

        resp = await async_client.get(f"/images/{image_id}")
        data = resp.json()
        assert data["caption"] == "A cat"
        assert data["ocr_text"] == "LOL"

    async def test_get_nonexistent_image_returns_404(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.get("/images/999999")
        assert resp.status_code == 404


class TestDeleteImage:
    async def test_delete_existing_image(
        self,
        async_client: AsyncClient,
        run_sync_seed,
        _patch_async_domain_session_scope: None,
    ) -> None:
        def seed(session) -> int:
            image = create_image(session=session)
            create_processing(session=session, image=image)
            create_annotation(session=session, image=image)
            return image.id

        image_id = await run_sync_seed(seed)

        resp = await async_client.delete(f"/images/{image_id}")
        assert resp.status_code == 204

    async def test_delete_nonexistent_image_returns_404(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.delete("/images/999999")
        assert resp.status_code == 404

    async def test_delete_calls_storage_delete(
        self,
        async_client: AsyncClient,
        run_sync_seed,
        api_storage,
        _patch_async_domain_session_scope: None,
    ) -> None:
        image_id = await run_sync_seed(
            lambda session: create_image(session=session, s3_key="images/test/abc.jpg").id
        )

        await async_client.delete(f"/images/{image_id}")
        assert api_storage.deleted == ["images/test/abc.jpg"]
