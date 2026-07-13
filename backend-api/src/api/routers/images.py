from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from api.auth import AdminRequired
from api.deps import (
    ArtifactStorageDep,
    MediaStorageDep,
    MediaUrlResolverDep,
    TemporalClientDep,
)
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
from domain.image_ingest_input import ImageIngestInput, RemoteImageUrlInput
from domain.image_upload import ImageUploadStager
from domain.job_store import ApiJobStore
from shared.config import settings
from workflows import IngestWorkflow, IngestWorkflowInput

router = APIRouter(prefix="/images", tags=["Images"], responses=error_responses(403, 429, 500))


async def _launch_ingest(
    temporal: TemporalClientDep,
    *,
    inputs: list[ImageIngestInput],
    dataset: str | None,
    tags: list[str],
    callback_url: str | None,
) -> ImageIngestResponse:
    store = ApiJobStore()
    job = await store.create_ingest_job(
        inputs=inputs, dataset=dataset, tags=tags, callback_url=callback_url
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
    await store.record_workflow_id(job.job_id, job.workflow_id)

    return ImageIngestResponse(
        job_id=job.job_id,
        queued=job.queued,
        duplicates=job.duplicates,
        message=f"Queued {job.queued} images for processing",
    )


@router.post("", response_model=ImageIngestResponse, status_code=202)
@limiter.limit(ADMIN_LIMIT)
async def ingest_images(
    request: Request,
    _auth: AdminRequired,
    ingest_request: ImageIngestRequest,
    temporal: TemporalClientDep,
) -> ImageIngestResponse:
    return await _launch_ingest(
        temporal,
        inputs=[RemoteImageUrlInput(url=str(url)) for url in ingest_request.urls],
        dataset=ingest_request.dataset,
        tags=ingest_request.tags,
        callback_url=str(ingest_request.callback_url) if ingest_request.callback_url else None,
    )


@router.post("/upload", response_model=ImageIngestResponse, status_code=202)
@limiter.limit(ADMIN_LIMIT)
async def upload_image(
    request: Request,
    _auth: AdminRequired,
    artifact_storage: ArtifactStorageDep,
    temporal: TemporalClientDep,
    file: Annotated[UploadFile, File()],
    dataset: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
) -> ImageIngestResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    staged = await ImageUploadStager(artifact_storage).stage(
        content=content, filename=file.filename, content_type=file.content_type
    )

    return await _launch_ingest(
        temporal,
        inputs=[staged],
        dataset=dataset,
        tags=tags or [],
        callback_url=None,
    )


@router.get("", response_model=ImageListResponse)
@limiter.limit(ADMIN_LIMIT)
async def list_images(
    request: Request,
    _auth: AdminRequired,
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: MediaUrlResolverDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[ImageStatus | None, Query()] = None,
    dataset: Annotated[str | None, Query()] = None,
    sort: Annotated[Literal["newest", "oldest"], Query()] = "newest",
) -> ImageListResponse:
    page = await ImageCatalog(media_storage, artifact_storage, media_urls).list_images(
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
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: MediaUrlResolverDep,
) -> ImageResponse:
    try:
        image = await ImageCatalog(media_storage, artifact_storage, media_urls).get_image(image_id)
    except ImageCatalogNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    payload = image.model_dump()
    payload["status"] = ImageStatus(payload["status"])
    return ImageResponse.model_construct(**payload)


@router.delete("/{image_id}", status_code=204, responses=error_responses(404))
async def delete_image(
    _auth: AdminRequired,
    image_id: int,
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: MediaUrlResolverDep,
) -> None:
    try:
        await ImageCatalog(media_storage, artifact_storage, media_urls).delete_image(image_id)
    except ImageCatalogNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
