from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from shared.models import ProcessingStatus
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


def _failed_run_item(session: Session):
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
    return source, run, item, url


def test_retry_run_starts_workflow_and_resets_failed(
    client: TestClient, db_session: Session, mock_temporal: AsyncMock
) -> None:
    source, run, _item, url = _failed_run_item(db_session)

    response = client.post(f"/sources/{source.id}/runs/{run.id}/retry")

    assert response.status_code == 202
    body = response.json()
    assert body["queued"] == 1
    assert body["workflow_id"] == f"source-retry-workflow-{body['job_id']}"
    mock_temporal.start_workflow.assert_called_once()
    db_session.refresh(url)
    assert url.status == ProcessingStatus.PENDING


def test_retry_run_errors(client: TestClient, db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)
    run = create_source_run(session=db_session, source=source)

    assert client.post(f"/sources/{source.id}/runs/{run.id}/retry").status_code == 409
    assert client.post(f"/sources/{source.id}/runs/999999/retry").status_code == 404
    assert client.post("/sources/999999/runs/1/retry").status_code == 404


def test_retry_source_starts_workflow_and_resets_failed(
    client: TestClient, db_session: Session, mock_temporal: AsyncMock
) -> None:
    source, _run, _item, url = _failed_run_item(db_session)

    response = client.post(f"/sources/{source.id}/retry")

    assert response.status_code == 202
    assert response.json()["queued"] == 1
    mock_temporal.start_workflow.assert_called_once()
    db_session.refresh(url)
    assert url.status == ProcessingStatus.PENDING


def test_retry_source_errors(client: TestClient, db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)

    assert client.post(f"/sources/{source.id}/retry").status_code == 409
    assert client.post("/sources/999999/retry").status_code == 404


def test_retry_item_starts_workflow_and_resets_failed(
    client: TestClient, db_session: Session, mock_temporal: AsyncMock
) -> None:
    source, _run, item, url = _failed_run_item(db_session)

    response = client.post(f"/sources/{source.id}/items/{item.id}/retry")

    assert response.status_code == 202
    assert response.json()["queued"] == 1
    mock_temporal.start_workflow.assert_called_once()
    db_session.refresh(url)
    assert url.status == ProcessingStatus.PENDING


def test_retry_item_errors(client: TestClient, db_session: Session) -> None:
    source = create_ingestion_source(session=db_session)

    assert client.post(f"/sources/{source.id}/items/999999/retry").status_code == 404
