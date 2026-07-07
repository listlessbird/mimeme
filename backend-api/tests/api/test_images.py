"""Tests for the /images endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from shared.config import settings
from shared.models.orm import IngestURL, Job, JobType, ProcessingStatus
from tests.factories import (
    create_annotation,
    create_image,
    create_processing,
)


class TestIngestImages:
    def test_ingest_creates_job_and_urls(self, client: TestClient, db_session: Session) -> None:
        resp = client.post(
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

        job = db_session.query(Job).filter_by(id=data["job_id"]).first()
        assert job is not None
        assert job.type == JobType.INGEST

        urls = db_session.query(IngestURL).filter_by(job_id=data["job_id"]).all()
        assert len(urls) == 2

    def test_ingest_deduplicates_urls(self, client: TestClient) -> None:
        resp = client.post(
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

    def test_ingest_starts_temporal_workflow(
        self, client: TestClient, mock_temporal: MagicMock
    ) -> None:
        resp = client.post(
            "/images",
            json={"urls": ["https://example.com/img1.jpg"]},
        )
        assert resp.status_code == 202
        mock_temporal.start_workflow.assert_called_once()

    def test_ingest_empty_urls_returns_422(self, client: TestClient) -> None:
        resp = client.post("/images", json={"urls": []})
        assert resp.status_code == 422

    def test_ingest_invalid_url_returns_422(self, client: TestClient) -> None:
        resp = client.post("/images", json={"urls": ["not-a-url"]})
        assert resp.status_code == 422


class TestUploadImage:
    def test_upload_stores_file_and_creates_ingest_job(
        self, client: TestClient, db_session: Session, api_storage
    ) -> None:
        resp = client.post(
            "/images/upload",
            files={"file": ("meme.jpg", b"fake-image-bytes", "image/jpeg")},
            data={"dataset": "memes", "tags": ["funny", "cats"]},
        )

        assert resp.status_code == 202
        data = resp.json()
        assert data["queued"] == 1
        assert data["duplicates"] == 0

        # stored to a staging key
        assert len(api_storage.uploaded) == 1
        stored_key = api_storage.uploaded[0].key
        assert stored_key.startswith("uploads/staging/")

        # converged on the URL-based ingest path: one ingest_url over the staged URL
        urls = db_session.query(IngestURL).filter_by(job_id=data["job_id"]).all()
        assert len(urls) == 1
        assert urls[0].url == "https://mock-s3/presigned"

    def test_upload_starts_workflow(self, client: TestClient, mock_temporal: MagicMock) -> None:
        resp = client.post(
            "/images/upload",
            files={"file": ("meme.png", b"bytes", "image/png")},
        )
        assert resp.status_code == 202
        mock_temporal.start_workflow.assert_called_once()

    def test_upload_empty_file_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/images/upload",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_upload_rejects_non_admin(
        self, client: TestClient, api_storage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "app_env", "production")
        monkeypatch.setattr(settings, "api_key_admin", "secret-admin-key")

        resp = client.post(
            "/images/upload",
            files={"file": ("meme.jpg", b"bytes", "image/jpeg")},
        )

        assert resp.status_code == 403
        assert api_storage.uploaded == []


class TestListImages:
    def test_list_images_empty(self, client: TestClient) -> None:
        resp = client.get("/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"] == []
        assert data["total"] == 0

    def test_list_images_with_data(self, client: TestClient, db_session: Session) -> None:
        create_image(session=db_session)
        create_image(session=db_session)
        db_session.flush()

        resp = client.get("/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["images"]) == 2

    def test_list_images_pagination(self, client: TestClient, db_session: Session) -> None:
        for _ in range(5):
            create_image(session=db_session)
        db_session.flush()

        resp = client.get("/images?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 2
        assert data["total"] == 5
        assert data["has_more"] is True

    def test_list_images_offset_beyond_total(self, client: TestClient, db_session: Session) -> None:
        create_image(session=db_session)
        db_session.flush()

        resp = client.get("/images?offset=999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"] == []
        assert data["total"] == 1
        assert data["has_more"] is False

    def test_list_images_filter_by_dataset(self, client: TestClient, db_session: Session) -> None:
        create_image(session=db_session, dataset="cats")
        create_image(session=db_session, dataset="dogs")
        db_session.flush()

        resp = client.get("/images?dataset=cats")
        data = resp.json()
        assert data["total"] == 1
        assert data["images"][0]["dataset"] == "cats"

    def test_list_images_filter_by_status_done(
        self, client: TestClient, db_session: Session
    ) -> None:
        img1 = create_image(session=db_session)
        create_processing(session=db_session, image=img1, embed_status=ProcessingStatus.DONE)
        img2 = create_image(session=db_session)
        create_processing(session=db_session, image=img2, embed_status=ProcessingStatus.PENDING)
        db_session.flush()

        resp = client.get("/images?status=done")
        data = resp.json()
        assert data["total"] == 1

    def test_list_images_sort_oldest(self, client: TestClient, db_session: Session) -> None:
        create_image(session=db_session)
        create_image(session=db_session)
        db_session.flush()

        resp = client.get("/images?sort=oldest")
        data = resp.json()
        assert data["images"][0]["id"] <= data["images"][1]["id"]


class TestGetImage:
    def test_get_existing_image(self, client: TestClient, db_session: Session) -> None:
        image = create_image(session=db_session)
        db_session.flush()

        resp = client.get(f"/images/{image.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == image.id
        assert data["sha256"] == image.sha256

    def test_get_image_with_annotations(self, client: TestClient, db_session: Session) -> None:
        image = create_image(session=db_session)
        create_annotation(session=db_session, image=image, caption_text="A cat", ocr_text="LOL")
        db_session.flush()

        resp = client.get(f"/images/{image.id}")
        data = resp.json()
        assert data["caption"] == "A cat"
        assert data["ocr_text"] == "LOL"

    def test_get_nonexistent_image_returns_404(self, client: TestClient) -> None:
        resp = client.get("/images/999999")
        assert resp.status_code == 404


class TestDeleteImage:
    def test_delete_existing_image(self, client: TestClient, db_session: Session) -> None:
        image = create_image(session=db_session)
        create_processing(session=db_session, image=image)
        create_annotation(session=db_session, image=image)
        db_session.flush()

        resp = client.delete(f"/images/{image.id}")
        assert resp.status_code == 204

    def test_delete_nonexistent_image_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/images/999999")
        assert resp.status_code == 404

    def test_delete_calls_storage_delete(
        self, client: TestClient, db_session: Session, api_storage
    ) -> None:
        image = create_image(session=db_session, s3_key="images/test/abc.jpg")
        db_session.flush()

        client.delete(f"/images/{image.id}")
        assert api_storage.deleted == ["images/test/abc.jpg"]
