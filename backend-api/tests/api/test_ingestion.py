"""Tests for the admin /ingestion endpoints (issue 07 — throughput lens)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from shared.config import settings
from shared.models import IngestStage, ProcessingStatus, SourceRunTrigger
from shared.models.orm import DuplicateReason
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


def _attempt(db: Session, **kwargs: object):
    job = kwargs.pop("job", None) or create_job(session=db)
    return create_ingest_url(session=db, job=job, **kwargs)


class TestIngestionList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get("/ingestion", params={"view": "all"})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"rows": [], "total": 0, "limit": 50, "offset": 0}

    def test_row_carries_grouping_and_derived_fields(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session, name="r/memes")
        run = create_source_run(
            session=db_session, source=src, trigger_mode=SourceRunTrigger.SCHEDULED
        )
        job = create_job(session=db_session)
        image = create_image(session=db_session, dataset="memes")
        attempt = _attempt(
            db_session,
            job=job,
            source_id=src.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            image_id=image.id,
        )
        db_session.flush()

        resp = client.get("/ingestion", params={"view": "all"})
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert len(rows) == 1
        row = rows[0]
        assert row["ingest_url_id"] == attempt.id
        assert row["job_id"] == job.id
        assert row["source_run_id"] == run.id
        assert row["source_id"] == src.id
        assert row["source_name"] == "r/memes"
        assert row["trigger"] == "scheduled"
        assert row["stage"] == "COMPLETE"
        assert row["outcome"] == "ingested"
        assert row["resolved_image_id"] == image.id
        assert row["dataset"] == "memes"
        assert row["thumbnail_url"] == "https://mock-s3/presigned"

    def test_manual_upload_has_manual_trigger_and_no_source(
        self, client: TestClient, db_session: Session
    ) -> None:
        _attempt(db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.DOWNLOADING)
        db_session.flush()

        rows = client.get("/ingestion", params={"view": "all"}).json()["rows"]
        assert rows[0]["trigger"] == "manual"
        assert rows[0]["source_id"] is None
        assert rows[0]["outcome"] == "in_flight"

    def test_deduped_outcome_and_canonical_target(
        self, client: TestClient, db_session: Session
    ) -> None:
        canonical = create_image(session=db_session)
        _attempt(
            db_session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.DEDUPED,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=canonical.id,
        )
        db_session.flush()

        row = client.get("/ingestion", params={"view": "all"}).json()["rows"][0]
        assert row["outcome"] == "deduped"
        assert row["duplicate_reason"] == "PHASH"
        assert row["resolved_image_id"] == canonical.id


class TestIngestionFilters:
    def test_stage_filter(self, client: TestClient, db_session: Session) -> None:
        _attempt(db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.ANNOTATING)
        _attempt(db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.EMBEDDING)
        db_session.flush()

        body = client.get("/ingestion", params={"view": "all", "stage": "ANNOTATING"}).json()
        assert body["total"] == 1
        assert body["rows"][0]["stage"] == "ANNOTATING"

    def test_trigger_filter(self, client: TestClient, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session, name="sched")
        run = create_source_run(
            session=db_session, source=src, trigger_mode=SourceRunTrigger.SCHEDULED
        )
        _attempt(db_session, source_id=src.id, source_run_id=run.id)
        _attempt(db_session)  # manual upload
        db_session.flush()

        sched = client.get("/ingestion", params={"view": "all", "trigger": "scheduled"}).json()
        manual = client.get("/ingestion", params={"view": "all", "trigger": "manual"}).json()
        assert sched["total"] == 1
        assert sched["rows"][0]["trigger"] == "scheduled"
        assert manual["total"] == 1
        assert manual["rows"][0]["trigger"] == "manual"

    def test_source_filter(self, client: TestClient, db_session: Session) -> None:
        a = create_ingestion_source(session=db_session, name="a")
        b = create_ingestion_source(session=db_session, name="b")
        _attempt(db_session, source_id=a.id)
        _attempt(db_session, source_id=b.id)
        db_session.flush()

        body = client.get("/ingestion", params={"view": "all", "source_id": a.id}).json()
        assert body["total"] == 1
        assert body["rows"][0]["source_id"] == a.id

    def test_dataset_filter(self, client: TestClient, db_session: Session) -> None:
        img = create_image(session=db_session, dataset="cats")
        _attempt(db_session, status=ProcessingStatus.DONE, image_id=img.id)
        other = create_image(session=db_session, dataset="dogs")
        _attempt(db_session, status=ProcessingStatus.DONE, image_id=other.id)
        db_session.flush()

        body = client.get("/ingestion", params={"view": "all", "dataset": "cats"}).json()
        assert body["total"] == 1
        assert body["rows"][0]["dataset"] == "cats"

    def test_outcome_filter(self, client: TestClient, db_session: Session) -> None:
        img = create_image(session=db_session)
        _attempt(db_session, status=ProcessingStatus.DONE, image_id=img.id)
        canonical = create_image(session=db_session)
        _attempt(
            db_session,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.SHA256,
            duplicate_of_image_id=canonical.id,
        )
        db_session.flush()

        ingested = client.get("/ingestion", params={"view": "all", "outcome": "ingested"}).json()
        deduped = client.get("/ingestion", params={"view": "all", "outcome": "deduped"}).json()
        assert ingested["total"] == 1
        assert ingested["rows"][0]["outcome"] == "ingested"
        assert deduped["total"] == 1
        assert deduped["rows"][0]["outcome"] == "deduped"

    def test_date_range_filter(self, client: TestClient, db_session: Session) -> None:
        old = _attempt(db_session, created_at=datetime(2026, 1, 1, tzinfo=UTC))
        new = _attempt(db_session, created_at=datetime(2026, 6, 1, tzinfo=UTC))
        db_session.flush()

        body = client.get(
            "/ingestion",
            params={"view": "all", "created_from": "2026-05-01T00:00:00Z"},
        ).json()
        assert body["total"] == 1
        assert body["rows"][0]["ingest_url_id"] == new.id
        assert old.id != new.id

    def test_paging_under_filter(self, client: TestClient, db_session: Session) -> None:
        for _ in range(5):
            _attempt(db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.PROCESSING)
        db_session.flush()

        page = client.get(
            "/ingestion",
            params={"view": "all", "stage": "PROCESSING", "limit": 2, "offset": 0},
        ).json()
        assert page["total"] == 5
        assert len(page["rows"]) == 2

    def test_limit_capped_at_100(self, client: TestClient) -> None:
        assert client.get("/ingestion", params={"limit": 1000}).status_code == 422


class TestIngestionViews:
    def test_live_includes_inflight_and_recent_excludes_old(
        self, client: TestClient, db_session: Session
    ) -> None:
        now = datetime.now(UTC)
        inflight = _attempt(
            db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.DOWNLOADING
        )
        recent = _attempt(
            db_session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            stage_updated_at=now - timedelta(minutes=1),
        )
        old = _attempt(
            db_session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            stage_updated_at=now - timedelta(hours=2),
        )
        db_session.flush()

        rows = client.get("/ingestion", params={"view": "live"}).json()["rows"]
        ids = {r["ingest_url_id"] for r in rows}
        assert inflight.id in ids
        assert recent.id in ids
        assert old.id not in ids

    def test_completed_view_only_terminal_success(
        self, client: TestClient, db_session: Session
    ) -> None:
        img = create_image(session=db_session)
        done = _attempt(db_session, status=ProcessingStatus.DONE, image_id=img.id)
        _attempt(db_session, status=ProcessingStatus.RUNNING, stage=IngestStage.EMBEDDING)
        _attempt(db_session, status=ProcessingStatus.FAILED, stage=IngestStage.ANNOTATING)
        db_session.flush()

        body = client.get("/ingestion", params={"view": "completed"}).json()
        assert body["total"] == 1
        assert body["rows"][0]["ingest_url_id"] == done.id

    def test_failed_view_surfaces_frozen_stage_and_error(
        self, client: TestClient, db_session: Session
    ) -> None:
        _attempt(
            db_session,
            status=ProcessingStatus.FAILED,
            stage=IngestStage.ANNOTATING,
            error_message="annotate boom",
        )
        db_session.flush()

        body = client.get("/ingestion", params={"view": "failed"}).json()
        assert body["total"] == 1
        row = body["rows"][0]
        assert row["status"] == "FAILED"
        assert row["stage"] == "ANNOTATING"
        assert row["outcome"] == "failed"
        assert row["error_message"] == "annotate boom"


class TestIngestionDetail:
    def test_detail_returns_attempt_scoped_fields(
        self, client: TestClient, db_session: Session
    ) -> None:
        src = create_ingestion_source(session=db_session, name="src")
        run = create_source_run(
            session=db_session, source=src, trigger_mode=SourceRunTrigger.MANUAL
        )
        item = create_source_item(session=db_session, source=src)
        image = create_image(session=db_session, dataset="memes")
        attempt = _attempt(
            db_session,
            source_id=src.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            image_id=image.id,
        )
        db_session.flush()

        resp = client.get(f"/ingestion/{attempt.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ingest_url_id"] == attempt.id
        assert body["image_id"] == image.id
        assert body["resolved_image_id"] == image.id
        assert body["trigger"] == "manual"
        assert body["source_id"] == src.id
        assert body["stage"] == "COMPLETE"

    def test_detail_failed_attempt(self, client: TestClient, db_session: Session) -> None:
        attempt = _attempt(
            db_session,
            status=ProcessingStatus.FAILED,
            stage=IngestStage.EMBEDDING,
            error_message="embed boom",
        )
        db_session.flush()

        body = client.get(f"/ingestion/{attempt.id}").json()
        assert body["status"] == "FAILED"
        assert body["stage"] == "EMBEDDING"
        assert body["error_message"] == "embed boom"
        assert body["image_id"] is None

    def test_detail_missing_returns_404(self, client: TestClient) -> None:
        assert client.get("/ingestion/999999").status_code == 404


class TestIngestionLogs:
    def test_logs_unavailable_when_axiom_not_configured(
        self,
        client: TestClient,
        db_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "axiom_api_token", "")
        monkeypatch.setattr(settings, "axiom_dataset", "")
        attempt = _attempt(db_session)
        db_session.flush()

        resp = client.get(f"/ingestion/{attempt.id}/logs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert body["entries"] == []

    def test_logs_missing_attempt_returns_404(self, client: TestClient) -> None:
        assert client.get("/ingestion/999999/logs").status_code == 404


class TestIngestionAdminGating:
    def test_endpoints_reject_non_admin(
        self, client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempt = _attempt(db_session)
        db_session.flush()

        monkeypatch.setattr(settings, "app_env", "production")
        monkeypatch.setattr(settings, "api_key_admin", "secret-admin-key")

        assert client.get("/ingestion").status_code == 403
        assert client.get(f"/ingestion/{attempt.id}").status_code == 403
        assert client.get(f"/ingestion/{attempt.id}/logs").status_code == 403
