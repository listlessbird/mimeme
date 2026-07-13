from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
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


async def test_create_returns_201_and_persists(
    async_client: AsyncClient,
    async_db_session: AsyncSession,
    _patch_async_domain_session_scope: None,
) -> None:
    response = await async_client.post("/sources", json=_VALID_BODY)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "r/memes hourly"
    assert data["enabled"] is True
    assert await async_db_session.get(IngestionSource, data["id"]) is not None


async def test_create_validates_adapter_and_duplicate_name(
    async_client: AsyncClient,
    _patch_async_domain_session_scope: None,
) -> None:
    invalid = await async_client.post("/sources", json={**_VALID_BODY, "adapter_key": "nope"})
    assert invalid.status_code == 400

    assert (await async_client.post("/sources", json=_VALID_BODY)).status_code == 201
    duplicate = await async_client.post("/sources", json=_VALID_BODY)
    assert duplicate.status_code == 409


async def test_list_empty_preserves_response_shape(
    async_client: AsyncClient,
    _patch_async_domain_session_scope: None,
) -> None:
    response = await async_client.get("/sources")

    assert response.status_code == 200
    assert response.json() == {"sources": [], "total": 0}


async def test_list_returns_stats_and_excludes_deleted(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session, name="visible")
        create_source_run(session=session, source=source)
        job = create_job(session=session)
        image = create_image(session=session)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=image.id,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            status=ProcessingStatus.FAILED,
        )
        deleted = create_ingestion_source(session=session, name="gone")
        return source.id, deleted.id

    source_id, deleted_id = await run_sync_seed(seed)
    assert (await async_client.delete(f"/sources/{deleted_id}")).status_code == 204

    response = await async_client.get("/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["sources"][0]["id"] == source_id
    assert data["sources"][0]["stats"] == {
        "run_count": 1,
        "items_discovered": 0,
        "duplicate_count": 1,
        "images_ingested": 1,
        "failed_count": 1,
    }


async def test_detail_carries_derived_run_counts(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, str]:
        source = create_ingestion_source(session=session)
        job = create_job(session=session)
        run = create_source_run(
            session=session,
            source=source,
            status=SourceRunStatus.RUNNING,
            ingest_job_id=job.id,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.FAILED,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
        )
        for _ in range(4):
            create_source_item(session=session, source=source, last_source_run_id=run.id)
        return source.id, job.id

    source_id, job_id = await run_sync_seed(seed)
    response = await async_client.get(f"/sources/{source_id}")

    assert response.status_code == 200
    run = response.json()["recent_runs"][0]
    assert run["status"] == SourceRunStatus.RUNNING.value
    assert run["ingest_job_id"] == job_id
    assert (run["discovered"], run["queued"], run["duplicate"], run["failed"]) == (
        4,
        3,
        1,
        1,
    )


async def test_get_patch_and_delete_missing_return_404(
    async_client: AsyncClient,
    _patch_async_domain_session_scope: None,
) -> None:
    assert (await async_client.get("/sources/999999")).status_code == 404
    assert (await async_client.patch("/sources/999999", json={"enabled": False})).status_code == 404
    assert (await async_client.delete("/sources/999999")).status_code == 404


async def test_patch_updates_only_provided_fields(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(
        lambda session: (
            create_ingestion_source(
                session=session, schedule_cron="0 * * * *", dataset="memes", enabled=True
            ).id
        )
    )

    response = await async_client.patch(
        f"/sources/{source_id}", json={"schedule_cron": "0 0 * * *", "enabled": False}
    )

    assert response.status_code == 200
    assert response.json()["schedule_cron"] == "0 0 * * *"
    assert response.json()["enabled"] is False
    assert response.json()["dataset"] == "memes"


async def test_delete_soft_deletes_and_hides_source(
    async_client: AsyncClient,
    async_db_session: AsyncSession,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(
        lambda session: create_ingestion_source(session=session, name="to-delete").id
    )

    assert (await async_client.delete(f"/sources/{source_id}")).status_code == 204
    row = await async_db_session.get(IngestionSource, source_id)
    assert row is not None and row.deleted_at is not None
    assert (await async_client.get("/sources")).json()["total"] == 0
    assert (await async_client.get(f"/sources/{source_id}")).status_code == 404


async def test_manual_run_starts_workflow(
    async_client: AsyncClient,
    run_sync_seed,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)

    response = await async_client.post(f"/sources/{source_id}/run")

    assert response.status_code == 202
    assert response.json()["workflow_id"].startswith(f"source-sync-manual-{source_id}-")
    mock_temporal.start_workflow.assert_called_once()


async def test_manual_run_rejects_missing_source(
    async_client: AsyncClient,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    assert (await async_client.post("/sources/999999/run")).status_code == 404
    mock_temporal.start_workflow.assert_not_called()


async def test_manual_run_rejects_soft_deleted_source(
    async_client: AsyncClient,
    run_sync_seed,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)
    assert (await async_client.delete(f"/sources/{source_id}")).status_code == 204

    assert (await async_client.post(f"/sources/{source_id}/run")).status_code == 404
    mock_temporal.start_workflow.assert_not_called()


async def test_source_items_preserve_metadata_paging_and_state_filters(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, dict[str, int]]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        job = create_job(session=session)
        ingested_image = create_image(session=session)
        ingested = create_source_item(
            session=session,
            source=source,
            external_item_id="abc123",
            title="A funny meme",
            raw_metadata={"subreddit": "memes", "preview": "https://prev/ingested.jpg"},
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=ingested.id,
            url="https://media/ingested.jpg",
            status=ProcessingStatus.DONE,
            image_id=ingested_image.id,
        )
        dedup_image = create_image(session=session)
        deduped = create_source_item(session=session, source=source)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=deduped.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
            duplicate_of_image_id=dedup_image.id,
        )
        failed = create_source_item(
            session=session,
            source=source,
            raw_metadata={"preview": "https://prev/failed.jpg"},
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=failed.id,
            status=ProcessingStatus.FAILED,
            error_message="download failed",
        )
        running = create_source_item(session=session, source=source)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=running.id,
            status=ProcessingStatus.RUNNING,
        )
        unknown = create_source_item(session=session, source=source)
        return source.id, {
            "ingested": ingested.id,
            "deduped": deduped.id,
            "failed": failed.id,
            "running": running.id,
            "unknown": unknown.id,
            "run": run.id,
            "ingested_image": ingested_image.id,
        }

    source_id, ids = await run_sync_seed(seed)
    response = await async_client.get(f"/sources/{source_id}/items?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert data["state_counts"] == {
        "ingested": 1,
        "deduped": 1,
        "failed": 1,
        "in_flight": 1,
        "unknown": 1,
    }
    by_id = {item["id"]: item for item in data["items"]}
    ingested = by_id[ids["ingested"]]
    assert ingested["external_item_id"] == "abc123"
    assert ingested["title"] == "A funny meme"
    assert ingested["raw_metadata"]["subreddit"] == "memes"
    assert ingested["ingest_state"] == "ingested"
    assert ingested["resolved_image_id"] == ids["ingested_image"]
    assert ingested["attempt_source_run_id"] == ids["run"]
    assert ingested["media_url"] == "https://media/ingested.jpg"
    assert ingested["thumbnail_url"] == "https://mock-s3/presigned"
    assert ingested["first_seen_at"] is not None
    assert ingested["last_seen_at"] is not None
    assert by_id[ids["deduped"]]["ingest_state"] == "deduped"
    assert by_id[ids["failed"]]["attempt_error_message"] == "download failed"
    assert by_id[ids["failed"]]["thumbnail_url"] == "https://prev/failed.jpg"
    assert by_id[ids["running"]]["ingest_state"] == "in_flight"
    assert by_id[ids["unknown"]]["ingest_state"] == "unknown"

    failed = await async_client.get(
        f"/sources/{source_id}/items", params={"status": "failed", "limit": 10}
    )
    assert failed.json()["total"] == 1
    assert [item["id"] for item in failed.json()["items"]] == [ids["failed"]]
    assert failed.json()["state_counts"] == data["state_counts"]

    first = await async_client.get(f"/sources/{source_id}/items?limit=2&offset=0")
    second = await async_client.get(f"/sources/{source_id}/items?limit=2&offset=2")
    assert (len(first.json()["items"]), first.json()["limit"], first.json()["offset"]) == (
        2,
        2,
        0,
    )
    assert (len(second.json()["items"]), second.json()["offset"]) == (2, 2)
    assert (await async_client.get("/sources/999999/items")).status_code == 404


async def test_run_items_preserve_status_thumbnail_paging_and_errors(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int, int, int]:
        source = create_ingestion_source(session=session, name="owner")
        other = create_ingestion_source(session=session, name="other")
        run = create_source_run(session=session, source=source)
        foreign_run = create_source_run(session=session, source=other)
        job = create_job(session=session)
        image = create_image(session=session)
        item = create_source_item(session=session, source=source)
        attempt = create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        for _ in range(2):
            create_ingest_url(session=session, job=job, source_id=source.id, source_run_id=run.id)
        return source.id, run.id, foreign_run.id, attempt.id

    source_id, run_id, foreign_run_id, attempt_id = await run_sync_seed(seed)
    first = await async_client.get(f"/sources/{source_id}/runs/{run_id}/items?limit=2&offset=0")
    second = await async_client.get(f"/sources/{source_id}/runs/{run_id}/items?limit=2&offset=2")

    assert first.status_code == 200
    assert first.json()["total"] == 3
    assert (len(first.json()["items"]), first.json()["limit"], first.json()["offset"]) == (
        2,
        2,
        0,
    )
    assert (len(second.json()["items"]), second.json()["offset"]) == (1, 2)
    attempt = next(item for item in first.json()["items"] if item["id"] == attempt_id)
    assert attempt["status"] == ProcessingStatus.DONE.value
    assert attempt["duplicate_reason"] is None
    assert attempt["thumbnail_url"] == "https://mock-s3/presigned"
    assert (await async_client.get("/sources/999999/runs/1/items")).status_code == 404
    assert (
        await async_client.get(f"/sources/{source_id}/runs/{foreign_run_id}/items")
    ).status_code == 404


async def test_all_source_endpoints_reject_non_admin(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key_admin", "secret-admin-key")
    source_id = 1

    responses = [
        await async_client.post("/sources", json=_VALID_BODY),
        await async_client.get("/sources"),
        await async_client.get(f"/sources/{source_id}"),
        await async_client.patch(f"/sources/{source_id}", json={"enabled": False}),
        await async_client.post(f"/sources/{source_id}/run"),
        await async_client.post(f"/sources/{source_id}/retry"),
        await async_client.post(f"/sources/{source_id}/runs/1/retry"),
        await async_client.post(f"/sources/{source_id}/items/1/retry"),
        await async_client.get(f"/sources/{source_id}/items"),
        await async_client.get(f"/sources/{source_id}/runs/1/items"),
        await async_client.delete(f"/sources/{source_id}"),
    ]

    assert [response.status_code for response in responses] == [403] * len(responses)
