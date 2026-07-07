from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from api.auth import AdminRequired
from api.deps import DbSession, StorageDep
from api.models.errors import error_responses
from api.models.ingestion import (
    IngestionDetailResponse,
    IngestionListResponse,
    IngestionLogEntryResponse,
    IngestionLogsResponse,
    IngestionRowResponse,
)
from api.services.ingestion_logs import AxiomLogReader
from domain.ingestion_browse import (
    AttemptNotFoundError,
    IngestionBrowser,
    IngestionView,
    IngestOutcome,
)
from shared.models import IngestStage, IngestURL, SourceRunTrigger

router = APIRouter(
    prefix="/ingestion", tags=["Ingestion"], responses=error_responses(403, 429, 500)
)


@router.get("", response_model=IngestionListResponse)
async def list_ingestion(
    _auth: AdminRequired,
    storage: StorageDep,
    view: Annotated[IngestionView, Query()] = IngestionView.LIVE,
    stage: Annotated[IngestStage | None, Query()] = None,
    trigger: Annotated[SourceRunTrigger | None, Query()] = None,
    source_id: Annotated[int | None, Query(ge=1)] = None,
    dataset: Annotated[str | None, Query(max_length=255)] = None,
    outcome: Annotated[IngestOutcome | None, Query()] = None,
    created_from: Annotated[datetime | None, Query()] = None,
    created_to: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> IngestionListResponse:
    page = IngestionBrowser(storage).list_attempts(
        limit=limit,
        offset=offset,
        view=view,
        stage=stage,
        trigger=trigger,
        source_id=source_id,
        dataset=dataset,
        outcome=outcome,
        created_from=created_from,
        created_to=created_to,
    )

    return IngestionListResponse(
        rows=[IngestionRowResponse.model_validate(row.model_dump()) for row in page.rows],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/{ingest_url_id}",
    response_model=IngestionDetailResponse,
    responses=error_responses(404),
)
async def get_ingestion_attempt(
    _auth: AdminRequired,
    ingest_url_id: int,
    storage: StorageDep,
) -> IngestionDetailResponse:
    try:
        detail = IngestionBrowser(storage).get_attempt(ingest_url_id)
    except AttemptNotFoundError:
        raise HTTPException(status_code=404, detail="Ingest attempt not found")

    return IngestionDetailResponse.model_validate(detail.model_dump())


@router.get(
    "/{ingest_url_id}/logs",
    response_model=IngestionLogsResponse,
    responses=error_responses(404),
)
async def get_ingestion_logs(
    _auth: AdminRequired,
    ingest_url_id: int,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> IngestionLogsResponse:
    row = db.execute(
        select(IngestURL.job_id, IngestURL.created_at).where(IngestURL.id == ingest_url_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingest attempt not found")

    job_id, created_at = row
    reader = AxiomLogReader()
    entries = reader.fetch_attempt_logs(
        ingest_url_id=ingest_url_id,
        job_id=job_id,
        created_at=created_at,
        limit=limit,
    )

    workflow_id = next((e.workflow_id for e in entries if e.workflow_id), None)
    return IngestionLogsResponse(
        available=reader.available,
        workflow_id=workflow_id,
        entries=[
            IngestionLogEntryResponse(
                time=e.time,
                level=e.level,
                event=e.event,
                activity_name=e.activity_name,
                step=e.step,
                outcome=e.outcome,
                duration_ms=e.duration_ms,
                error=e.error,
                attempt=e.attempt,
                workflow_id=e.workflow_id,
            )
            for e in entries
        ],
    )
