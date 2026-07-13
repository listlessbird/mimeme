from __future__ import annotations

from unittest.mock import AsyncMock

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from shared.models import ProcessingStatus
from shared.models.orm import IngestURL
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


def _failed_run_item(session: Session) -> tuple[int, int, int, int]:
    source = create_ingestion_source(session=session)
    run = create_source_run(session=session, source=source)
    item = create_source_item(session=session, source=source)
    job = create_job(session=session)
    url = create_ingest_url(
        session=session,
        job=job,
        source_id=source.id,
        source_run_id=run.id,
        source_item_id=item.id,
        status=ProcessingStatus.FAILED,
        error_message="boom",
    )
    return source.id, run.id, item.id, url.id


async def test_retry_run_starts_workflow_and_resets_failed(
    async_client: AsyncClient,
    async_db_session: AsyncSession,
    run_sync_seed,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id, run_id, _item_id, url_id = await run_sync_seed(_failed_run_item)

    response = await async_client.post(f"/sources/{source_id}/runs/{run_id}/retry")

    assert response.status_code == 202
    body = response.json()
    assert body["queued"] == 1
    assert body["workflow_id"] == f"source-retry-workflow-{body['job_id']}"
    mock_temporal.start_workflow.assert_called_once()
    url = await async_db_session.get(IngestURL, url_id)
    assert url is not None and url.status == ProcessingStatus.PENDING


async def test_retry_run_errors(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        return source.id, run.id

    source_id, run_id = await run_sync_seed(seed)
    assert (await async_client.post(f"/sources/{source_id}/runs/{run_id}/retry")).status_code == 409
    assert (await async_client.post(f"/sources/{source_id}/runs/999999/retry")).status_code == 404
    assert (await async_client.post("/sources/999999/runs/1/retry")).status_code == 404


async def test_retry_source_starts_workflow_and_resets_failed(
    async_client: AsyncClient,
    async_db_session: AsyncSession,
    run_sync_seed,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id, _run_id, _item_id, url_id = await run_sync_seed(_failed_run_item)

    response = await async_client.post(f"/sources/{source_id}/retry")

    assert response.status_code == 202
    assert response.json()["queued"] == 1
    mock_temporal.start_workflow.assert_called_once()
    url = await async_db_session.get(IngestURL, url_id)
    assert url is not None and url.status == ProcessingStatus.PENDING


async def test_retry_source_errors(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)

    assert (await async_client.post(f"/sources/{source_id}/retry")).status_code == 409
    assert (await async_client.post("/sources/999999/retry")).status_code == 404


async def test_retry_item_starts_workflow_and_resets_failed(
    async_client: AsyncClient,
    async_db_session: AsyncSession,
    run_sync_seed,
    mock_temporal: AsyncMock,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id, _run_id, item_id, url_id = await run_sync_seed(_failed_run_item)

    response = await async_client.post(f"/sources/{source_id}/items/{item_id}/retry")

    assert response.status_code == 202
    assert response.json()["queued"] == 1
    mock_temporal.start_workflow.assert_called_once()
    url = await async_db_session.get(IngestURL, url_id)
    assert url is not None and url.status == ProcessingStatus.PENDING


async def test_retry_item_errors(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)

    assert (await async_client.post(f"/sources/{source_id}/items/999999/retry")).status_code == 404
