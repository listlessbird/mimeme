from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select

from api.auth import AdminRequired
from api.deps import DbSession, StorageDep, TemporalClientDep
from api.models.images import (
    ImageIngestRequest,
    ImageIngestResponse,
    ImageListResponse,
    ImageResponse,
    ImageStatus,
)
from api.rate_limit import ADMIN_LIMIT, limiter
from shared.config import settings
from shared.models import Annotation, Artifact, IngestURL, Job, JobType, Processing
from shared.models import ORMImage as Image
from workflows import IngestWorkflow, IngestWorkflowInput

router = APIRouter()


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
    unique_urls = list(dict.fromkeys(urls))
    duplicates = len(urls) - len(unique_urls)

    job_id = f"ingest-{uuid.uuid4().hex[:12]}"

    job = Job(id=job_id, type=JobType.INGEST)
    db.add(job)
    db.flush()

    for url in unique_urls:
        ingest_url = IngestURL(job_id=job_id, url=url)
        db.add(ingest_url)

    db.commit()

    await temporal.start_workflow(
        IngestWorkflow.run,
        IngestWorkflowInput(
            job_id=job_id,
            dataset=ingest_request.dataset,
            tags=ingest_request.tags,
            callback_url=str(ingest_request.callback_url) if ingest_request.callback_url else None,
        ),
        id=f"ingest-workflow-{job_id}",
        task_queue=settings.temporal_task_queue_cpu,
    )

    return ImageIngestResponse(
        job_id=job_id,
        queued=len(unique_urls),
        duplicates=duplicates,
        message=f"Queued {len(unique_urls)} images for processing",
    )


@router.get("", response_model=ImageListResponse)
@limiter.limit(ADMIN_LIMIT)
async def list_images(
    request: Request,
    _auth: AdminRequired,
    db: DbSession,
    storage: StorageDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: ImageStatus | None = Query(default=None),
    dataset: str | None = Query(default=None),
    sort: Literal["newest", "oldest"] = Query(default="newest"),
) -> ImageListResponse:
    query = select(Image)

    if dataset:
        query = query.where(Image.dataset == dataset)

    if status:
        query = query.join(Processing, Image.id == Processing.image_id, isouter=True)

        if status == ImageStatus.DONE:
            query = query.where(Processing.embed_status == "done")
        elif status == ImageStatus.PENDING:
            query = query.where(
                (Processing.embed_status == "pending") | (Processing.embed_status.is_(None))
            )
        elif status == ImageStatus.FAILED:
            query = query.where(
                (Processing.ocr_status == "failed")
                | (Processing.caption_status == "failed")
                | (Processing.embed_status == "failed")
            )

    count_query = select(func.count()).select_from(query.subquery())
    total = db.execute(count_query).scalar() or 0

    if sort == "newest":
        query = query.order_by(Image.id.desc())
    else:
        query = query.order_by(Image.id.asc())

    query = query.limit(limit).offset(offset)
    images = db.execute(query).scalars().all()

    results: list[ImageResponse] = []
    for img in images:
        proc = db.query(Processing).filter_by(image_id=img.id).first()
        ann = db.query(Annotation).filter_by(image_id=img.id).first()

        img_status = _compute_status(proc)

        url = None
        if img.s3_key:
            url = storage.generate_presigned_url(
                img.s3_key, expiration=settings.s3_presigned_url_expiry
            )

        results.append(
            ImageResponse(
                id=img.id,
                sha256=img.sha256,
                url=url,
                s3_key=img.s3_key,
                dataset=img.dataset,
                width=img.width,
                height=img.height,
                format=img.format,
                phash=img.phash,
                status=img_status,
                ocr_status=proc.ocr_status.value if proc else None,
                caption_status=proc.caption_status.value if proc else None,
                embed_status=proc.embed_status.value if proc else None,
                caption=ann.caption_text if ann else None,
                ocr_text=ann.ocr_text if ann else None,
                tags=[],
                created_at=img.created_at,
            )
        )

    return ImageListResponse(
        images=results,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(results)) < total,
    )


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    _auth: AdminRequired,
    image_id: int,
    db: DbSession,
    storage: StorageDep,
) -> ImageResponse:
    img = db.query(Image).filter_by(id=image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    proc = db.query(Processing).filter_by(image_id=img.id).first()
    ann = db.query(Annotation).filter_by(image_id=img.id).first()

    img_status = _compute_status(proc)

    url = None
    if img.s3_key:
        url = storage.generate_presigned_url(
            img.s3_key, expiration=settings.s3_presigned_url_expiry
        )

    return ImageResponse(
        id=img.id,
        sha256=img.sha256,
        url=url,
        s3_key=img.s3_key,
        dataset=img.dataset,
        width=img.width,
        height=img.height,
        format=img.format,
        phash=img.phash,
        status=img_status,
        ocr_status=proc.ocr_status.value if proc else None,
        caption_status=proc.caption_status.value if proc else None,
        embed_status=proc.embed_status.value if proc else None,
        caption=ann.caption_text if ann else None,
        ocr_text=ann.ocr_text if ann else None,
        tags=[],
        created_at=img.created_at,
    )


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    _auth: AdminRequired,
    image_id: int,
    db: DbSession,
    storage: StorageDep,
) -> None:
    img = db.query(Image).filter_by(id=image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    if img.s3_key:
        try:
            storage.delete(img.s3_key)
        except Exception:
            pass

    proc = db.query(Processing).filter_by(image_id=image_id).first()
    if proc and proc.embed_s3_key:
        try:
            storage.delete(proc.embed_s3_key)
            text_key = proc.embed_s3_key.replace(".npy", "_text.npy")
            storage.delete(text_key)
        except Exception:
            pass

    db.query(Annotation).filter_by(image_id=image_id).delete()
    db.query(Processing).filter_by(image_id=image_id).delete()
    db.query(Artifact).filter_by(image_id=image_id).delete()
    db.delete(img)
    db.commit()


def _compute_status(proc: Processing | None) -> ImageStatus:
    if not proc:
        return ImageStatus.PENDING

    if proc.embed_status.value == "done":
        return ImageStatus.DONE

    if any(s.value == "failed" for s in [proc.ocr_status, proc.caption_status, proc.embed_status]):
        return ImageStatus.FAILED

    if proc.embed_status.value == "running":
        return ImageStatus.EMBEDDING

    if proc.caption_status.value == "running" or proc.ocr_status.value == "running":
        return ImageStatus.ANNOTATING

    return ImageStatus.PENDING
