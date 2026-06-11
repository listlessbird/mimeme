from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.auth import AdminRequired
from api.deps import DbSession
from api.models.sources import (
    CreateSourceRequest,
    SourceDetailResponse,
    SourceListItemResponse,
    SourceListResponse,
    SourceResponse,
    UpdateSourceRequest,
)
from domain.source_registry import (
    DuplicateSourceNameError,
    SourceNotFoundError,
    SourceRegistry,
    UnknownAdapterKeyError,
)

router = APIRouter(prefix="/sources", tags=["Sources"])


@router.post("", response_model=SourceResponse, status_code=201)
async def create_source(
    _auth: AdminRequired, body: CreateSourceRequest, db: DbSession
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
    _auth: AdminRequired, source_id: int, body: UpdateSourceRequest, db: DbSession
) -> SourceResponse:
    try:
        view = SourceRegistry(db).patch(source_id, **body.model_dump(exclude_unset=True))
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceResponse.model_validate(view.model_dump())


@router.delete("/{source_id}", status_code=204)
async def delete_source(_auth: AdminRequired, source_id: int, db: DbSession) -> None:
    try:
        SourceRegistry(db).soft_delete(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
