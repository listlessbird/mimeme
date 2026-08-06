from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query

from mimeme.api.auth import AdminRequired
from mimeme.api.deps import DbDep, SettingsDep, TemporalClientDep, UrlsDep
from mimeme.api.models.errors import error_responses
from mimeme.api.models.sources import (
    CreateSourceRequest,
    RetryResponse,
    RunItemListResponse,
    RunItemResponse,
    SourceDetailResponse,
    SourceItemListResponse,
    SourceItemResponse,
    SourceListItemResponse,
    SourceListResponse,
    SourceResponse,
    TriggerRunResponse,
    UpdateSourceRequest,
)
from mimeme.config import Settings
from mimeme.db import Db
from mimeme.db.schema import SourceRunTrigger
from mimeme.source import (
    DuplicateSourceName,
    NothingToRetry,
    RetryInput,
    RetryPlan,
    RunNotFound,
    SourceItemIngestState,
    SourceItemNotFound,
    SourceNotFound,
    SyncInput,
    UnknownAdapterKey,
    rule,
    schedule,
    store,
)
from mimeme.source import retry as source_retry
from mimeme.source.workflow import SourceRetryWorkflow, SourceSyncWorkflow

router = APIRouter(prefix="/sources", tags=["Sources"], responses=error_responses(403, 429, 500))
log = structlog.get_logger()


async def _sync_schedules(db: Db, temporal: TemporalClientDep) -> None:
    try:
        desired = await store.list_schedule_specs(db)
        await schedule.reconcile(schedule.TemporalScheduleStore(temporal), desired=desired)
    except Exception as exc:
        log.warning(
            "inline_schedule_sync_failed",
            error=str(exc),
            message="reconciliation will heal on next sweep",
        )


@router.post(
    "",
    response_model=SourceResponse,
    status_code=201,
    responses=error_responses(400, 409),
)
async def create_source(
    _auth: AdminRequired, db: DbDep, body: CreateSourceRequest, temporal: TemporalClientDep
) -> SourceResponse:
    try:
        view = await store.create(
            db,
            name=body.name,
            adapter_key=body.adapter_key,
            adapter_config=dict(body.adapter_config),
            dataset=body.dataset,
            schedule_cron=body.schedule_cron,
            schedule_timezone=body.schedule_timezone,
            max_items_per_run=body.max_items_per_run,
            enabled=body.enabled,
        )
    except UnknownAdapterKey as exc:
        raise HTTPException(status_code=400, detail=f"Unknown adapter_key: {exc}")
    except DuplicateSourceName as exc:
        raise HTTPException(status_code=409, detail=f"Source name already in use: {exc}")

    await _sync_schedules(db, temporal)

    return SourceResponse.model_validate(view.model_dump())


@router.get("", response_model=SourceListResponse)
async def list_sources(_auth: AdminRequired, db: DbDep) -> SourceListResponse:
    items = await store.list_sources(db)
    return SourceListResponse(
        sources=[SourceListItemResponse.model_validate(item.model_dump()) for item in items],
        total=len(items),
    )


@router.get("/{source_id}", response_model=SourceDetailResponse, responses=error_responses(404))
async def get_source(_auth: AdminRequired, db: DbDep, source_id: int) -> SourceDetailResponse:
    try:
        detail = await store.get_source(db, source_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceDetailResponse.model_validate(detail.model_dump())


@router.patch("/{source_id}", response_model=SourceResponse, responses=error_responses(404))
async def update_source(
    _auth: AdminRequired,
    db: DbDep,
    source_id: int,
    body: UpdateSourceRequest,
    temporal: TemporalClientDep,
) -> SourceResponse:
    try:
        patch = body.model_dump(exclude_unset=True)
        if patch.get("adapter_config") is None:
            patch.pop("adapter_config", None)
        view = await store.patch(db, source_id, patch)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")

    await _sync_schedules(db, temporal)

    return SourceResponse.model_validate(view.model_dump())


@router.post(
    "/{source_id}/run",
    response_model=TriggerRunResponse,
    status_code=202,
    responses=error_responses(404),
)
async def trigger_source_run(
    _auth: AdminRequired,
    db: DbDep,
    settings: SettingsDep,
    source_id: int,
    temporal: TemporalClientDep,
) -> TriggerRunResponse:
    try:
        await store.get_source(db, source_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")

    # A fresh id per manual trigger so it never collides with the scheduled
    # Source's deterministic schedule workflow id (issue 04).
    workflow_id = rule.manual_workflow_id(source_id, uuid.uuid4().hex[:12])
    await temporal.start_workflow(
        SourceSyncWorkflow.run,
        SyncInput(source_id=source_id, trigger=SourceRunTrigger.MANUAL),
        id=workflow_id,
        task_queue=settings.temporal.task_queue,
    )

    return TriggerRunResponse(workflow_id=workflow_id, message="Source run started")


async def _start_retry(
    settings: Settings, temporal: TemporalClientDep, plan: RetryPlan
) -> RetryResponse:
    await temporal.start_workflow(
        SourceRetryWorkflow.run,
        RetryInput(
            job_id=plan.job_id,
            source_run_ids=plan.source_run_ids,
            dataset=plan.dataset,
            items=plan.items,
        ),
        id=plan.workflow_id,
        task_queue=settings.temporal.task_queue,
    )
    return RetryResponse(
        job_id=plan.job_id,
        workflow_id=plan.workflow_id,
        queued=plan.count,
        message=f"Re-queued {plan.count} failed item(s)",
    )


@router.post(
    "/{source_id}/retry",
    response_model=RetryResponse,
    status_code=202,
    responses=error_responses(404, 409),
)
async def retry_source(
    _auth: AdminRequired,
    db: DbDep,
    settings: SettingsDep,
    source_id: int,
    temporal: TemporalClientDep,
) -> RetryResponse:
    try:
        plan = await source_retry.retry_source(db, source_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")
    except NothingToRetry:
        raise HTTPException(status_code=409, detail="No failed items to retry")

    return await _start_retry(settings, temporal, plan)


@router.post(
    "/{source_id}/runs/{run_id}/retry",
    response_model=RetryResponse,
    status_code=202,
    responses=error_responses(404, 409),
)
async def retry_source_run(
    _auth: AdminRequired,
    db: DbDep,
    settings: SettingsDep,
    source_id: int,
    run_id: int,
    temporal: TemporalClientDep,
) -> RetryResponse:
    try:
        plan = await source_retry.retry_run(db, source_id, run_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")
    except RunNotFound:
        raise HTTPException(status_code=404, detail="Run not found")
    except NothingToRetry:
        raise HTTPException(status_code=409, detail="No failed items to retry")

    return await _start_retry(settings, temporal, plan)


@router.post(
    "/{source_id}/items/{item_id}/retry",
    response_model=RetryResponse,
    status_code=202,
    responses=error_responses(404, 409),
)
async def retry_source_item(
    _auth: AdminRequired,
    db: DbDep,
    settings: SettingsDep,
    source_id: int,
    item_id: int,
    temporal: TemporalClientDep,
) -> RetryResponse:
    try:
        plan = await source_retry.retry_item(db, source_id, item_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")
    except SourceItemNotFound:
        raise HTTPException(status_code=404, detail="Source item not found")
    except NothingToRetry:
        raise HTTPException(status_code=409, detail="No failed attempt to retry")

    return await _start_retry(settings, temporal, plan)


@router.get(
    "/{source_id}/items",
    response_model=SourceItemListResponse,
    responses=error_responses(404),
)
async def list_source_items(
    _auth: AdminRequired,
    db: DbDep,
    source_id: int,
    media_urls: UrlsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: SourceItemIngestState | None = None,
) -> SourceItemListResponse:
    try:
        page = await store.list_items(
            db, source_id, media_urls, limit=limit, offset=offset, status=status
        )
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceItemListResponse(
        items=[SourceItemResponse.model_validate(item.model_dump()) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        state_counts=page.state_counts,
    )


@router.get(
    "/{source_id}/runs/{run_id}/items",
    response_model=RunItemListResponse,
    responses=error_responses(404),
)
async def list_run_items(
    _auth: AdminRequired,
    db: DbDep,
    source_id: int,
    run_id: int,
    media_urls: UrlsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunItemListResponse:
    try:
        page = await store.list_run_items(
            db, source_id, run_id, media_urls, limit=limit, offset=offset
        )
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")
    except RunNotFound:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunItemListResponse(
        items=[RunItemResponse.model_validate(item.model_dump()) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.delete("/{source_id}", status_code=204, responses=error_responses(404))
async def delete_source(
    _auth: AdminRequired, db: DbDep, source_id: int, temporal: TemporalClientDep
) -> None:
    try:
        await store.soft_delete(db, source_id)
    except SourceNotFound:
        raise HTTPException(status_code=404, detail="Source not found")

    await _sync_schedules(db, temporal)
