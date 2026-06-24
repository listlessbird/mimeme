"""Tests for the admin /sources endpoints (issue 02, slice D)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from shared.config import settings
from shared.models import IngestionSource, ProcessingStatus, SourceRunStatus
from shared.models.orm import DuplicateReason
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

_VALID_BODY = {
    "name": "r/memes hourly",
    "adapter_key": "meme_api",
    "adapter_config": {"subreddits": ["memes"]},
    "dataset": "memes",
    "schedule_cron": "0 * * * *",
    "schedule_timezone": "UTC",
    "max_items_per_run": 50,
}


class TestCreateSource:
    def test_create_returns_201_and_persists(self, client: TestClient, db_session: Session) -> None:
        resp = client.post("/sources", json=_VALID_BODY)

        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "r/memes hourly"
        assert data["enabled"] is True
        assert db_session.query(IngestionSource).filter_by(id=data["id"]).count() == 1

    def test_create_unknown_adapter_key_returns_400(self, client: TestClient) -> None:
        resp = client.post("/sources", json={**_VALID_BODY, "adapter_key": "nope"})

        assert resp.status_code == 400

    def test_create_duplicate_live_name_returns_409(self, client: TestClient) -> None:
        client.post("/sources", json=_VALID_BODY)

        resp = client.post("/sources", json=_VALID_BODY)

        assert resp.status_code == 409


class TestListSources:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/sources")

        assert resp.status_code == 200
        assert resp.json() == {"sources": [], "total": 0}

    def test_list_returns_stats_and_excludes_soft_deleted(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session, name="visible")
        create_source_run(session=db_session, source=src)
        gone = create_ingestion_source(session=db_session, name="gone")
        client.delete(f"/sources/{gone.id}")

        data = client.get("/sources").json()

        names = {s["name"] for s in data["sources"]}
        assert names == {"visible"}
        assert data["total"] == 1
        assert data["sources"][0]["stats"]["run_count"] == 1


class TestGetSource:
    def test_detail_carries_derived_run_counts(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src, status=SourceRunStatus.RUNNING)
        job = create_job(session=db_session)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            status=ProcessingStatus.FAILED,
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
        )
        for _ in range(4):
            create_source_item(session=db_session, source=src, last_source_run_id=run.id)

        data = client.get(f"/sources/{src.id}").json()

        assert data["id"] == src.id
        run_view = data["recent_runs"][0]
        assert run_view["status"] == SourceRunStatus.RUNNING.value
        assert (run_view["discovered"], run_view["queued"]) == (4, 3)
        assert (run_view["duplicate"], run_view["failed"]) == (1, 1)

    def test_missing_returns_404(self, client: TestClient) -> None:
        assert client.get("/sources/999999").status_code == 404

    def test_soft_deleted_returns_404(self, client: TestClient, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        assert client.delete(f"/sources/{src.id}").status_code == 204

        assert client.get(f"/sources/{src.id}").status_code == 404


class TestPatchSource:
    def test_patch_updates_only_provided_fields(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(
            session=db_session, schedule_cron="0 * * * *", dataset="memes", enabled=True
        )

        resp = client.patch(
            f"/sources/{src.id}", json={"schedule_cron": "0 0 * * *", "enabled": False}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["schedule_cron"] == "0 0 * * *"
        assert data["enabled"] is False
        assert data["dataset"] == "memes"  # untouched

    def test_patch_missing_returns_404(self, client: TestClient) -> None:
        assert client.patch("/sources/999999", json={"enabled": False}).status_code == 404


class TestDeleteSource:
    def test_delete_soft_deletes_and_hides_from_listing(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session, name="to-delete")

        assert client.delete(f"/sources/{src.id}").status_code == 204

        row = db_session.query(IngestionSource).filter_by(id=src.id).one()
        assert row.deleted_at is not None
        assert client.get("/sources").json()["total"] == 0

    def test_delete_missing_returns_404(self, client: TestClient) -> None:
        assert client.delete("/sources/999999").status_code == 404


class TestTriggerSourceRun:
    def test_run_starts_workflow_and_returns_202(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)

        resp = client.post(f"/sources/{src.id}/run")

        assert resp.status_code == 202
        body = resp.json()
        assert body["workflow_id"].startswith(f"source-sync-manual-{src.id}-")
        mock_temporal.start_workflow.assert_called_once()

    def test_run_unknown_source_returns_404(self, client: TestClient) -> None:
        resp = client.post("/sources/999999/run")

        assert resp.status_code == 404

    def test_run_soft_deleted_source_returns_404(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)
        client.delete(f"/sources/{src.id}")

        resp = client.post(f"/sources/{src.id}/run")

        assert resp.status_code == 404
        mock_temporal.start_workflow.assert_not_called()


class TestListSourceItems:
    def test_returns_items_with_metadata_and_preview(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)
        create_source_item(
            session=db_session,
            source=src,
            external_item_id="abc123",
            title="A funny meme",
            raw_metadata={
                "subreddit": "memes",
                "author": "u/someone",
                "ups": 4200,
                "postLink": "https://reddit.com/r/memes/comments/abc123/x",
                "preview": "https://preview.redd.it/abc123.jpg",
            },
        )

        data = client.get(f"/sources/{src.id}/items").json()

        assert data["total"] == 1
        item = data["items"][0]
        assert item["external_item_id"] == "abc123"
        assert item["title"] == "A funny meme"
        assert item["raw_metadata"]["subreddit"] == "memes"
        assert item["thumbnail_url"] == "https://preview.redd.it/abc123.jpg"
        assert item["first_seen_at"] is not None
        assert item["last_seen_at"] is not None

    def test_paging(self, client: TestClient, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        for _ in range(3):
            create_source_item(session=db_session, source=src)

        first = client.get(f"/sources/{src.id}/items?limit=2&offset=0").json()
        second = client.get(f"/sources/{src.id}/items?limit=2&offset=2").json()

        assert first["total"] == 3
        assert (len(first["items"]), first["limit"], first["offset"]) == (2, 2, 0)
        assert (len(second["items"]), second["offset"]) == (1, 2)

    def test_missing_source_returns_404(self, client: TestClient) -> None:
        assert client.get("/sources/999999/items").status_code == 404


class TestListRunItems:
    def test_attempts_carry_status_dedup_and_thumbnail(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src)
        job = create_job(session=db_session)

        ingested_img = create_image(session=db_session)
        ingested_item = create_source_item(
            session=db_session, source=src, raw_metadata={"preview": "https://prev/ingested.jpg"}
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=ingested_item.id,
            status=ProcessingStatus.DONE,
            image_id=ingested_img.id,
        )

        sha_img = create_image(session=db_session)
        sha_item = create_source_item(session=db_session, source=src)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=sha_item.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.SHA256,
            image_id=sha_img.id,
            duplicate_of_image_id=sha_img.id,
        )

        phash_img = create_image(session=db_session)
        phash_item = create_source_item(session=db_session, source=src)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=phash_item.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
            image_id=phash_img.id,
            duplicate_of_image_id=phash_img.id,
        )

        failed_item = create_source_item(
            session=db_session, source=src, raw_metadata={"preview": "https://prev/failed.jpg"}
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=failed_item.id,
            status=ProcessingStatus.FAILED,
        )

        data = client.get(f"/sources/{src.id}/runs/{run.id}/items").json()

        assert data["total"] == 4
        ingested, sha, phash, failed = data["items"]

        assert ingested["status"] == ProcessingStatus.DONE.value
        assert ingested["duplicate_reason"] is None
        assert ingested["image_id"] == ingested_img.id
        # resolved to an image with an s3_key -> presigned (mock_storage)
        assert ingested["thumbnail_url"] == "https://mock-s3/presigned"

        assert sha["duplicate_reason"] == DuplicateReason.SHA256.value
        assert sha["image_id"] == sha_img.id
        assert phash["duplicate_reason"] == DuplicateReason.PHASH.value
        assert phash["image_id"] == phash_img.id

        assert failed["status"] == ProcessingStatus.FAILED.value
        assert failed["image_id"] is None
        # no resolved image -> falls back to the upstream preview
        assert failed["thumbnail_url"] == "https://prev/failed.jpg"

    def test_paging(self, client: TestClient, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src)
        job = create_job(session=db_session)
        for _ in range(3):
            create_ingest_url(
                session=db_session, job=job, source_id=src.id, source_run_id=run.id
            )

        first = client.get(f"/sources/{src.id}/runs/{run.id}/items?limit=2&offset=0").json()
        second = client.get(f"/sources/{src.id}/runs/{run.id}/items?limit=2&offset=2").json()

        assert first["total"] == 3
        assert (len(first["items"]), first["limit"], first["offset"]) == (2, 2, 0)
        assert (len(second["items"]), second["offset"]) == (1, 2)

    def test_missing_source_returns_404(self, client: TestClient) -> None:
        assert client.get("/sources/999999/runs/1/items").status_code == 404

    def test_foreign_run_returns_404(self, client: TestClient, db_session: Session) -> None:
        owner = create_ingestion_source(session=db_session, name="owner")
        other = create_ingestion_source(session=db_session, name="other")
        run = create_source_run(session=db_session, source=other)

        assert client.get(f"/sources/{owner.id}/runs/{run.id}/items").status_code == 404


class TestAdminOnly:
    """Non-admin callers are rejected on every Source endpoint.

    The test client forces APP_ENV=development (auth bypassed), so flip the
    live setting to production with no API key and assert each route 403s.
    """

    def test_all_endpoints_reject_non_admin(
        self, client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = create_ingestion_source(session=db_session)
        monkeypatch.setattr(settings, "app_env", "production")
        monkeypatch.setattr(settings, "api_key_admin", "secret-admin-key")

        assert client.post("/sources", json=_VALID_BODY).status_code == 403
        assert client.get("/sources").status_code == 403
        assert client.get(f"/sources/{src.id}").status_code == 403
        assert client.patch(f"/sources/{src.id}", json={"enabled": False}).status_code == 403
        assert client.post(f"/sources/{src.id}/run").status_code == 403
        assert client.get(f"/sources/{src.id}/items").status_code == 403
        assert client.get(f"/sources/{src.id}/runs/1/items").status_code == 403
        assert client.delete(f"/sources/{src.id}").status_code == 403
