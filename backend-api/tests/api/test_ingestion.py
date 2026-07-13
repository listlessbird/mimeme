from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
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


async def test_list_empty(
    async_client: AsyncClient, _patch_async_domain_session_scope: None
) -> None:
    response = await async_client.get("/ingestion", params={"view": "all"})

    assert response.status_code == 200
    assert response.json() == {"rows": [], "total": 0, "limit": 50, "offset": 0}


async def test_list_row_carries_grouping_and_derived_fields(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, str, int, int, int]:
        source = create_ingestion_source(session=session, name="r/memes")
        run = create_source_run(
            session=session, source=source, trigger_mode=SourceRunTrigger.SCHEDULED
        )
        job = create_job(session=session)
        image = create_image(session=session, dataset="memes")
        attempt = _attempt(
            session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            image_id=image.id,
        )
        return attempt.id, job.id, source.id, run.id, image.id

    attempt_id, job_id, source_id, run_id, image_id = await run_sync_seed(seed)
    response = await async_client.get("/ingestion", params={"view": "all"})

    assert response.status_code == 200
    row = response.json()["rows"][0]
    assert row["ingest_url_id"] == attempt_id
    assert row["job_id"] == job_id
    assert row["source_run_id"] == run_id
    assert row["source_id"] == source_id
    assert row["source_name"] == "r/memes"
    assert row["trigger"] == "scheduled"
    assert row["stage"] == "COMPLETE"
    assert row["outcome"] == "ingested"
    assert row["resolved_image_id"] == image_id
    assert row["dataset"] == "memes"
    assert row["thumbnail_url"] == "https://mock-s3/presigned"


async def test_list_derives_manual_and_deduped_outcomes(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> int:
        _attempt(session, status=ProcessingStatus.RUNNING, stage=IngestStage.DOWNLOADING)
        canonical = create_image(session=session)
        _attempt(
            session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.DEDUPED,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=canonical.id,
        )
        return canonical.id

    canonical_id = await run_sync_seed(seed)
    response = await async_client.get("/ingestion", params={"view": "all"})
    rows = response.json()["rows"]

    manual = next(row for row in rows if row["outcome"] == "in_flight")
    deduped = next(row for row in rows if row["outcome"] == "deduped")
    assert manual["trigger"] == "manual"
    assert manual["source_id"] is None
    assert deduped["duplicate_reason"] == "PHASH"
    assert deduped["resolved_image_id"] == canonical_id


async def test_filters_and_paging(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source_a = create_ingestion_source(session=session, name="a")
        source_b = create_ingestion_source(session=session, name="b")
        run = create_source_run(
            session=session, source=source_a, trigger_mode=SourceRunTrigger.SCHEDULED
        )
        image = create_image(session=session, dataset="cats")
        _attempt(
            session,
            source_id=source_a.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.ANNOTATING,
            image_id=image.id,
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        _attempt(
            session,
            source_id=source_b.id,
            status=ProcessingStatus.RUNNING,
            stage=IngestStage.EMBEDDING,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for _ in range(4):
            _attempt(
                session,
                status=ProcessingStatus.RUNNING,
                stage=IngestStage.PROCESSING,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        return source_a.id, source_b.id

    source_a_id, _source_b_id = await run_sync_seed(seed)

    checks = [
        ({"stage": "ANNOTATING"}, "stage", "ANNOTATING"),
        ({"trigger": "scheduled"}, "trigger", "scheduled"),
        ({"source_id": source_a_id}, "source_id", source_a_id),
        ({"dataset": "cats"}, "dataset", "cats"),
        ({"outcome": "ingested"}, "outcome", "ingested"),
        ({"created_from": "2026-05-01T00:00:00Z"}, "source_id", source_a_id),
    ]
    for params, key, value in checks:
        body = (await async_client.get("/ingestion", params={"view": "all", **params})).json()
        assert body["total"] == 1
        assert body["rows"][0][key] == value

    page = (
        await async_client.get(
            "/ingestion",
            params={"view": "all", "stage": "PROCESSING", "limit": 2, "offset": 0},
        )
    ).json()
    assert page["total"] == 4
    assert len(page["rows"]) == 2
    assert (await async_client.get("/ingestion", params={"limit": 1000})).status_code == 422


async def test_named_views_select_expected_attempts(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int, int, int]:
        now = datetime.now(UTC)
        inflight = _attempt(session, status=ProcessingStatus.RUNNING, stage=IngestStage.DOWNLOADING)
        image = create_image(session=session)
        done = _attempt(
            session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            stage_updated_at=now - timedelta(minutes=1),
            image_id=image.id,
        )
        old = _attempt(
            session,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            stage_updated_at=now - timedelta(hours=2),
            image_id=image.id,
        )
        failed = _attempt(
            session,
            status=ProcessingStatus.FAILED,
            stage=IngestStage.ANNOTATING,
            error_message="annotate boom",
        )
        return inflight.id, done.id, old.id, failed.id

    inflight_id, done_id, old_id, failed_id = await run_sync_seed(seed)

    live = (await async_client.get("/ingestion", params={"view": "live"})).json()["rows"]
    live_ids = {row["ingest_url_id"] for row in live}
    assert {inflight_id, done_id} <= live_ids
    assert old_id not in live_ids

    completed = (await async_client.get("/ingestion", params={"view": "completed"})).json()
    assert completed["total"] == 2
    failed = (await async_client.get("/ingestion", params={"view": "failed"})).json()
    assert failed["total"] == 1
    assert failed["rows"][0]["ingest_url_id"] == failed_id
    assert failed["rows"][0]["error_message"] == "annotate boom"


async def test_detail_returns_attempt_fields_and_missing_404(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int, int]:
        source = create_ingestion_source(session=session, name="src")
        run = create_source_run(
            session=session, source=source, trigger_mode=SourceRunTrigger.MANUAL
        )
        item = create_source_item(session=session, source=source)
        image = create_image(session=session, dataset="memes")
        attempt = _attempt(
            session,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            image_id=image.id,
        )
        return attempt.id, source.id, image.id

    attempt_id, source_id, image_id = await run_sync_seed(seed)
    response = await async_client.get(f"/ingestion/{attempt_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["image_id"] == image_id
    assert body["resolved_image_id"] == image_id
    assert body["trigger"] == "manual"
    assert body["source_id"] == source_id
    assert body["stage"] == "COMPLETE"
    assert (await async_client.get("/ingestion/999999")).status_code == 404


def test_logs_unavailable_when_axiom_not_configured(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "axiom_api_token", "")
    monkeypatch.setattr(settings, "axiom_dataset", "")
    attempt = _attempt(db_session)
    db_session.flush()

    response = client.get(f"/ingestion/{attempt.id}/logs")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["entries"] == []


def test_logs_missing_attempt_returns_404(client: TestClient) -> None:
    assert client.get("/ingestion/999999/logs").status_code == 404


def test_endpoints_reject_non_admin(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _attempt(db_session)
    db_session.flush()
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key_admin", "secret-admin-key")

    assert client.get("/ingestion").status_code == 403
    assert client.get(f"/ingestion/{attempt.id}").status_code == 403
    assert client.get(f"/ingestion/{attempt.id}/logs").status_code == 403
