from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query

from mimeme.activities.scheduling.reconcile import reconcile
from mimeme.activities.scheduling.temporal_store import TemporalScheduleStore
from mimeme.api.auth import AdminRequired
from mimeme.api.deps import DbDep, MediaUrlResolverDep, SettingsDep, TemporalClientDep
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
from mimeme.db.schema import SourceRunTrigger
from mimeme.domain.source_item_browse import (
    RunNotFoundError,
    SourceItemBrowser,
    SourceItemIngestState,
)
from mimeme.domain.source_registry import (
    DuplicateSourceNameError,
    SourceNotFoundError,
    SourceRegistry,
    UnknownAdapterKeyError,
)
from mimeme.domain.source_retry import (
    NothingToRetryError,
    RetryPlan,
    SourceItemNotFoundError,
    SourceRetry,
)
from mimeme.db import Db
from mimeme.shared.config import Settings
from mimeme.workflows import (
    SourceRetryWorkflow,
    SourceRetryWorkflowInput,
    SourceSyncWorkflow,
    SourceSyncWorkflowInput,
)

router = APIRouter(prefix="/sources", tags=["Sources"], responses=error_responses(403, 429, 500))
log = structlog.get_logger()


async def _sync_schedules(db: Db, temporal: TemporalClientDep) -> None:
    try:
        desired = await SourceRegistry(db).list_schedule_specs()
        await reconcile(TemporalScheduleStore(temporal), desired=desired)
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
        view = await SourceRegistry(db).create(
            name=body.name,
            adapter_key=body.adapter_key,
            adapter_config=dict(body.adapter_config),
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
async def list_sources(_auth: AdminRequired, db: DbDep) -> SourceListResponse:
    items = await SourceRegistry(db).list_sources()
    return SourceListResponse(
        sources=[SourceListItemResponse.model_validate(item.model_dump()) for item in items],
        total=len(items),
    )


@router.get("/{source_id}", response_model=SourceDetailResponse, responses=error_responses(404))
async def get_source(_auth: AdminRequired, db: DbDep, source_id: int) -> SourceDetailResponse:
    try:
        detail = await SourceRegistry(db).get_source(source_id)
    except SourceNotFoundError:
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
        view = await SourceRegistry(db).patch(source_id, **patch)
    except SourceNotFoundError:
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
        await SourceRegistry(db).get_source(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    # A fresh id per manual trigger so it never collides with the scheduled
    # Source's deterministic schedule workflow id (issue 04).
    workflow_id = f"source-sync-manual-{source_id}-{uuid.uuid4().hex[:12]}"
    await temporal.start_workflow(
        SourceSyncWorkflow.run,
        SourceSyncWorkflowInput(source_id=source_id, trigger=SourceRunTrigger.MANUAL),
        id=workflow_id,
        task_queue=settings.temporal.task_queue,
    )

    return TriggerRunResponse(workflow_id=workflow_id, message="Source run started")


async def _start_retry(
    settings: Settings, temporal: TemporalClientDep, plan: RetryPlan
) -> RetryResponse:
    await temporal.start_workflow(
        SourceRetryWorkflow.run,
        SourceRetryWorkflowInput(
            job_id=plan.job_id,
            source_run_ids=plan.source_run_ids,
            dataset=plan.dataset,
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
        plan = await SourceRetry(db).retry_source(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except NothingToRetryError:
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
        plan = await SourceRetry(db).retry_run(source_id, run_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    except NothingToRetryError:
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
        plan = await SourceRetry(db).retry_item(source_id, item_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except SourceItemNotFoundError:
        raise HTTPException(status_code=404, detail="Source item not found")
    except NothingToRetryError:
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
    media_urls: MediaUrlResolverDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: SourceItemIngestState | None = None,
) -> SourceItemListResponse:
    try:
        page = await SourceItemBrowser(db, media_urls).list_items(
            source_id, limit=limit, offset=offset, status=status
        )
    except SourceNotFoundError:
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
    media_urls: MediaUrlResolverDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunItemListResponse:
    try:
        page = await SourceItemBrowser(db, media_urls).list_run_items(
            source_id, run_id, limit=limit, offset=offset
        )
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunItemListResponse(
        items=[RunItemResponse.model_validate(item.model_dump()) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.delete("/{source_id}", status_code=204, responses=error_responses(404))
async def delete_source(_auth: AdminRequired, source_id: int, temporal: TemporalClientDep) -> None:
    try:
        await SourceRegistry().soft_delete(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    await _sync_schedules(temporal)
