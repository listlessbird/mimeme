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
        job = create_job(session=db_session)
        image = create_image(session=db_session)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=image.id,
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            status=ProcessingStatus.FAILED,
        )
        gone = create_ingestion_source(session=db_session, name="gone")
        client.delete(f"/sources/{gone.id}")

        data = client.get("/sources").json()

        names = {s["name"] for s in data["sources"]}
        assert names == {"visible"}
        assert data["total"] == 1
        assert data["sources"][0]["stats"]["run_count"] == 1
        assert data["sources"][0]["stats"]["duplicate_count"] == 1
        assert data["sources"][0]["stats"]["images_ingested"] == 1
        assert data["sources"][0]["stats"]["failed_count"] == 1


class TestGetSource:
    def test_detail_carries_derived_run_counts(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)
        job = create_job(session=db_session)
        image = create_image(session=db_session)
        run = create_source_run(
            session=db_session,
            source=src,
            status=SourceRunStatus.RUNNING,
            ingest_job_id=job.id,
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
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
        assert data["stats"]["images_ingested"] == 1
        run_view = data["recent_runs"][0]
        assert run_view["status"] == SourceRunStatus.RUNNING.value
        assert run_view["ingest_job_id"] == job.id
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


def _failed_run_item(db_session: Session, src):
    run = create_source_run(session=db_session, source=src, source_id=src.id)
    item = create_source_item(session=db_session, source=src, source_id=src.id)
    job = create_job(session=db_session)
    url = create_ingest_url(
        session=db_session,
        job=job,
        source_id=src.id,
        source_run_id=run.id,
        source_item_id=item.id,
        status=ProcessingStatus.FAILED,
        error_message="boom",
    )
    return run, item, url


class TestRetryRun:
    def test_retry_run_starts_workflow_and_resets_failed(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)
        run, _item, url = _failed_run_item(db_session, src)

        resp = client.post(f"/sources/{src.id}/runs/{run.id}/retry")

        assert resp.status_code == 202
        body = resp.json()
        assert body["queued"] == 1
        assert body["job_id"]
        assert body["workflow_id"] == f"source-retry-workflow-{body['job_id']}"
        mock_temporal.start_workflow.assert_called_once()
        db_session.refresh(url)
        assert url.status == ProcessingStatus.PENDING

    def test_retry_run_no_failures_returns_409(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src, source_id=src.id)

        resp = client.post(f"/sources/{src.id}/runs/{run.id}/retry")

        assert resp.status_code == 409
        mock_temporal.start_workflow.assert_not_called()

    def test_retry_run_unknown_run_returns_404(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)

        resp = client.post(f"/sources/{src.id}/runs/999999/retry")

        assert resp.status_code == 404

    def test_retry_run_unknown_source_returns_404(self, client: TestClient) -> None:
        resp = client.post("/sources/999999/runs/1/retry")

        assert resp.status_code == 404


class TestRetrySource:
    def test_retry_source_starts_workflow_and_resets_failed(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)
        _run, _item, url = _failed_run_item(db_session, src)

        resp = client.post(f"/sources/{src.id}/retry")

        assert resp.status_code == 202
        body = resp.json()
        assert body["queued"] == 1
        mock_temporal.start_workflow.assert_called_once()
        db_session.refresh(url)
        assert url.status == ProcessingStatus.PENDING

    def test_retry_source_no_failures_returns_409(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)

        resp = client.post(f"/sources/{src.id}/retry")

        assert resp.status_code == 409
        mock_temporal.start_workflow.assert_not_called()

    def test_retry_source_unknown_source_returns_404(self, client: TestClient) -> None:
        assert client.post("/sources/999999/retry").status_code == 404


class TestRetryItem:
    def test_retry_item_starts_workflow_and_resets_failed(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        src = create_ingestion_source(session=db_session)
        _run, item, url = _failed_run_item(db_session, src)

        resp = client.post(f"/sources/{src.id}/items/{item.id}/retry")

        assert resp.status_code == 202
        body = resp.json()
        assert body["queued"] == 1
        mock_temporal.start_workflow.assert_called_once()
        db_session.refresh(url)
        assert url.status == ProcessingStatus.PENDING

    def test_retry_item_unknown_item_returns_404(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)

        resp = client.post(f"/sources/{src.id}/items/999999/retry")

        assert resp.status_code == 404


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

    def test_items_carry_ingest_outcome_and_filter_server_side(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src)
        job = create_job(session=db_session)

        ingested_image = create_image(session=db_session)
        ingested_item = create_source_item(
            session=db_session, source=src, raw_metadata={"preview": "https://prev/ingested.jpg"}
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=ingested_item.id,
            url="https://media/ingested.jpg",
            status=ProcessingStatus.DONE,
            image_id=ingested_image.id,
        )

        dedup_image = create_image(session=db_session)
        dedup_item = create_source_item(
            session=db_session, source=src, raw_metadata={"preview": "https://prev/dedup.jpg"}
        )
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=dedup_item.id,
            url="https://media/dedup.jpg",
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=dedup_image.id,
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
            url="https://media/failed.jpg",
            status=ProcessingStatus.FAILED,
            error_message="download failed",
        )

        running_item = create_source_item(session=db_session, source=src)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=running_item.id,
            status=ProcessingStatus.RUNNING,
        )
        unknown_item = create_source_item(session=db_session, source=src)

        data = client.get(f"/sources/{src.id}/items?limit=10").json()

        assert data["total"] == 5
        assert data["state_counts"] == {
            "ingested": 1,
            "deduped": 1,
            "failed": 1,
            "in_flight": 1,
            "unknown": 1,
        }
        by_id = {item["id"]: item for item in data["items"]}

        assert by_id[ingested_item.id]["ingest_state"] == "ingested"
        assert by_id[ingested_item.id]["resolved_image_id"] == ingested_image.id
        assert by_id[ingested_item.id]["duplicate_reason"] is None
        assert by_id[ingested_item.id]["duplicate_of_image_id"] is None
        assert by_id[ingested_item.id]["attempt_status"] == ProcessingStatus.DONE.value
        assert by_id[ingested_item.id]["attempt_source_run_id"] == run.id
        assert by_id[ingested_item.id]["media_url"] == "https://media/ingested.jpg"
        assert by_id[ingested_item.id]["thumbnail_url"] == "https://mock-s3/presigned"

        assert by_id[dedup_item.id]["ingest_state"] == "deduped"
        assert by_id[dedup_item.id]["resolved_image_id"] == dedup_image.id
        assert by_id[dedup_item.id]["duplicate_reason"] == DuplicateReason.PHASH.value
        assert by_id[dedup_item.id]["duplicate_of_image_id"] == dedup_image.id
        assert by_id[dedup_item.id]["thumbnail_url"] == "https://mock-s3/presigned"

        assert by_id[failed_item.id]["ingest_state"] == "failed"
        assert by_id[failed_item.id]["resolved_image_id"] is None
        assert by_id[failed_item.id]["attempt_error_message"] == "download failed"
        assert by_id[failed_item.id]["thumbnail_url"] == "https://prev/failed.jpg"

        assert by_id[running_item.id]["ingest_state"] == "in_flight"
        assert by_id[unknown_item.id]["ingest_state"] == "unknown"

        failed = client.get(f"/sources/{src.id}/items?status=failed&limit=10").json()
        assert failed["total"] == 1
        assert [item["id"] for item in failed["items"]] == [failed_item.id]
        assert failed["state_counts"] == data["state_counts"]

        in_flight = client.get(f"/sources/{src.id}/items?status=in_flight&limit=1").json()
        assert in_flight["total"] == 1
        assert [item["id"] for item in in_flight["items"]] == [running_item.id]

        second_dedup_page = client.get(
            f"/sources/{src.id}/items?status=deduped&limit=1&offset=1"
        ).json()
        assert second_dedup_page["total"] == 1
        assert second_dedup_page["items"] == []


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
            error_message="download timed out",
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
        assert failed["error_message"] == "download timed out"
        assert failed["source_item_id"] == failed_item.id
        assert failed["image_id"] is None
        # no resolved image -> falls back to the upstream preview
        assert failed["thumbnail_url"] == "https://prev/failed.jpg"

    def test_paging(self, client: TestClient, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src)
        job = create_job(session=db_session)
        for _ in range(3):
            create_ingest_url(session=db_session, job=job, source_id=src.id, source_run_id=run.id)

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
