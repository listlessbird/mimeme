from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException

from activities.scheduling.reconcile import run_reconciliation
from activities.scheduling.temporal_store import TemporalScheduleStore
from api.auth import AdminRequired
from api.deps import DbSession, TemporalClientDep
from api.models.sources import (
    CreateSourceRequest,
    SourceDetailResponse,
    SourceListItemResponse,
    SourceListResponse,
    SourceResponse,
    TriggerRunResponse,
    UpdateSourceRequest,
)
from domain.source_registry import (
    DuplicateSourceNameError,
    SourceNotFoundError,
    SourceRegistry,
    UnknownAdapterKeyError,
)
from shared.config import settings
from shared.models import SourceRunTrigger
from workflows import SourceSyncWorkflow, SourceSyncWorkflowInput

router = APIRouter(prefix="/sources", tags=["Sources"])
log = structlog.get_logger()


async def _sync_schedules(db: DbSession, temporal: TemporalClientDep) -> None:
    try:
        await run_reconciliation(db, TemporalScheduleStore(temporal))
    except Exception as exc:
        log.warning(
            "inline_schedule_sync_failed",
            error=str(exc),
            message="reconciliation will heal on next sweep",
        )


@router.post("", response_model=SourceResponse, status_code=201)
async def create_source(
    _auth: AdminRequired, body: CreateSourceRequest, db: DbSession, temporal: TemporalClientDep
) -> SourceResponse:
    try:
        view = SourceRegistry(db).create(
            name=body.name,
            adapter_key=body.adapter_key,
            adapter_config=body.adapter_config,
            dataset=body.dataset,
            schedule_cron=body.schedule_cron,
            schedule_timezone=body.schedule_timezone,
            max_items_per_run=body.max_items_per_run,
            enabled=body.enabled,
        )
    except UnknownAdapterKeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown adapter_key: {exc}")
    except DuplicateSourceNameError as exc:
        raise HTTPException(status_code=409, detail=f"Source name already in use: {exc}")

    await _sync_schedules(db, temporal)

    return SourceResponse.model_validate(view.model_dump())


@router.get("", response_model=SourceListResponse)
async def list_sources(_auth: AdminRequired, db: DbSession) -> SourceListResponse:
    items = SourceRegistry(db).list_sources()
    return SourceListResponse(
        sources=[SourceListItemResponse.model_validate(item.model_dump()) for item in items],
        total=len(items),
    )


@router.get("/{source_id}", response_model=SourceDetailResponse)
async def get_source(_auth: AdminRequired, source_id: int, db: DbSession) -> SourceDetailResponse:
    try:
        detail = SourceRegistry(db).get_source(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceDetailResponse.model_validate(detail.model_dump())


@router.patch("/{source_id}", response_model=SourceResponse)
async def update_source(
    _auth: AdminRequired,
    source_id: int,
    body: UpdateSourceRequest,
    db: DbSession,
    temporal: TemporalClientDep,
) -> SourceResponse:
    try:
        view = SourceRegistry(db).patch(source_id, **body.model_dump(exclude_unset=True))
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    await _sync_schedules(db, temporal)

    return SourceResponse.model_validate(view.model_dump())


@router.post("/{source_id}/run", response_model=TriggerRunResponse, status_code=202)
async def trigger_source_run(
    _auth: AdminRequired, source_id: int, db: DbSession, temporal: TemporalClientDep
) -> TriggerRunResponse:
    try:
        SourceRegistry(db).get_source(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    # A fresh id per manual trigger so it never collides with the scheduled
    # Source's deterministic schedule workflow id (issue 04).
    workflow_id = f"source-sync-manual-{source_id}-{uuid.uuid4().hex[:12]}"
    await temporal.start_workflow(
        SourceSyncWorkflow.run,
        SourceSyncWorkflowInput(source_id=source_id, trigger=SourceRunTrigger.MANUAL),
        id=workflow_id,
        task_queue=settings.temporal_task_queue,
    )

    return TriggerRunResponse(workflow_id=workflow_id, message="Source run started")


@router.delete("/{source_id}", status_code=204)
async def delete_source(
    _auth: AdminRequired, source_id: int, db: DbSession, temporal: TemporalClientDep
) -> None:
    try:
        SourceRegistry(db).soft_delete(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    await _sync_schedules(db, temporal)
