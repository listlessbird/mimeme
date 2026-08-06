from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from mimeme.api.auth import AdminRequired
from mimeme.api.deps import (
    ArtifactStorageDep,
    DbDep,
    EnvDep,
    MediaStorageDep,
    UrlsDep,
)
from mimeme.api.models.errors import error_responses
from mimeme.api.models.images import (
    ImageIngestRequest,
    ImageIngestResponse,
    ImageListResponse,
    ImageResponse,
    ImageStatus,
)
from mimeme.api.rate_limit import ADMIN_LIMIT, limiter
from mimeme.env import Env
from mimeme.ingest import RemoteUrl, Source, Submission
from mimeme.ingest.catalog import Catalog, NotFound
from mimeme.ingest.submit import stage_upload, submit

router = APIRouter(prefix="/images", tags=["Images"], responses=error_responses(403, 429, 500))


async def _launch_ingest(
    env: Env,
    *,
    inputs: list[Source],
    dataset: str | None,
    tags: list[str],
    callback_url: str | None,
) -> ImageIngestResponse:
    job = await submit(
        env,
        Submission(
            urls=inputs,
            dataset=dataset,
            tags=tags,
            callback_url=callback_url,
        ),
    )

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
    env: EnvDep,
    ingest_request: ImageIngestRequest,
) -> ImageIngestResponse:
    return await _launch_ingest(
        env,
        inputs=[RemoteUrl(url=str(url)) for url in ingest_request.urls],
        dataset=ingest_request.dataset,
        tags=ingest_request.tags,
        callback_url=str(ingest_request.callback_url) if ingest_request.callback_url else None,
    )


@router.post("/upload", response_model=ImageIngestResponse, status_code=202)
@limiter.limit(ADMIN_LIMIT)
async def upload_image(
    request: Request,
    _auth: AdminRequired,
    env: EnvDep,
    file: Annotated[UploadFile, File()],
    dataset: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
) -> ImageIngestResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    staged = await stage_upload(
        env, content=content, filename=file.filename, content_type=file.content_type
    )

    return await _launch_ingest(
        env,
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
    db: DbDep,
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: UrlsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[ImageStatus | None, Query()] = None,
    dataset: Annotated[str | None, Query()] = None,
    sort: Annotated[Literal["newest", "oldest"], Query()] = "newest",
) -> ImageListResponse:
    page = await Catalog(db, media_storage, artifact_storage, media_urls).list_images(
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
    db: DbDep,
    image_id: int,
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: UrlsDep,
) -> ImageResponse:
    try:
        image = await Catalog(db, media_storage, artifact_storage, media_urls).get_image(image_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Image not found")
    payload = image.model_dump()
    payload["status"] = ImageStatus(payload["status"])
    return ImageResponse.model_construct(**payload)


@router.delete("/{image_id}", status_code=204, responses=error_responses(404))
async def delete_image(
    _auth: AdminRequired,
    db: DbDep,
    image_id: int,
    media_storage: MediaStorageDep,
    artifact_storage: ArtifactStorageDep,
    media_urls: UrlsDep,
) -> None:
    try:
        await Catalog(db, media_storage, artifact_storage, media_urls).delete_image(image_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Image not found")
