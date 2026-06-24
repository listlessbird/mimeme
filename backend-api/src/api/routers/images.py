from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from api.auth import AdminRequired
from api.deps import DbSession, StorageDep, TemporalClientDep
from api.models.errors import error_responses
from api.models.images import (
    ImageIngestRequest,
    ImageIngestResponse,
    ImageListResponse,
    ImageResponse,
    ImageStatus,
)
from api.rate_limit import ADMIN_LIMIT, limiter
from domain.image_catalog import ImageCatalog, ImageCatalogNotFoundError
from domain.job_lifecycle import JobLifecycle
from shared.config import settings
from workflows import IngestWorkflow, IngestWorkflowInput

router = APIRouter(prefix="/images", tags=["Images"], responses=error_responses(403, 429, 500))


@router.post("", response_model=ImageIngestResponse, status_code=202)
@limiter.limit(ADMIN_LIMIT)
async def ingest_images(
    request: Request,
    _auth: AdminRequired,
    ingest_request: ImageIngestRequest,
    db: DbSession,
    temporal: TemporalClientDep,
) -> ImageIngestResponse:
    urls = [str(url) for url in ingest_request.urls]
    lifecycle = JobLifecycle(db)
    job = lifecycle.create_ingest_job(
        urls=urls,
        dataset=ingest_request.dataset,
        tags=ingest_request.tags,
        callback_url=str(ingest_request.callback_url) if ingest_request.callback_url else None,
    )

    await temporal.start_workflow(
        IngestWorkflow.run,
        IngestWorkflowInput(
            job_id=job.job_id,
            dataset=job.dataset,
            tags=job.tags,
            callback_url=job.callback_url,
        ),
        id=job.workflow_id,
        task_queue=settings.temporal_task_queue,
    )
    lifecycle.record_workflow_id(job.job_id, job.workflow_id)

    return ImageIngestResponse(
        job_id=job.job_id,
        queued=job.queued,
        duplicates=job.duplicates,
        message=f"Queued {job.queued} images for processing",
    )


@router.get("", response_model=ImageListResponse)
@limiter.limit(ADMIN_LIMIT)
async def list_images(
    request: Request,
    _auth: AdminRequired,
    db: DbSession,
    storage: StorageDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[ImageStatus | None, Query()] = None,
    dataset: Annotated[str | None, Query()] = None,
    sort: Annotated[Literal["newest", "oldest"], Query()] = "newest",
) -> ImageListResponse:
    page = ImageCatalog(db, storage).list_images(
        limit=limit,
        offset=offset,
        status=status.value if status else None,
        dataset=dataset,
        sort=sort,
    )

    image_responses: list[ImageResponse] = []
    for image in page.images:
        payload = image.model_dump()
        payload["status"] = ImageStatus(payload["status"])
        image_responses.append(ImageResponse.model_construct(**payload))

    return ImageListResponse(
        images=image_responses,
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
    )


@router.get("/{image_id}", response_model=ImageResponse, responses=error_responses(404))
async def get_image(
    _auth: AdminRequired,
    image_id: int,
    db: DbSession,
    storage: StorageDep,
) -> ImageResponse:
    try:
        image = ImageCatalog(db, storage).get_image(image_id)
    except ImageCatalogNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    payload = image.model_dump()
    payload["status"] = ImageStatus(payload["status"])
    return ImageResponse.model_construct(**payload)


@router.delete("/{image_id}", status_code=204, responses=error_responses(404))
async def delete_image(
    _auth: AdminRequired,
    image_id: int,
    db: DbSession,
    storage: StorageDep,
) -> None:
    try:
        ImageCatalog(db, storage).delete_image(image_id)
    except ImageCatalogNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
