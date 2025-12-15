import uuid
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query
from prometheus_client import Counter
from sqlalchemy import func, select

from api.config import settings
from api.deps import DbSession, StorageDep
from api.models.images import (
    ImageIngestRequest,
    ImageIngestResponse,
    ImageListResponse,
    ImageResponse,
    ImageStatus,
)
from api.models.orm import Annotation, Artifact, Processing
from api.models.orm import Image as ORMImage
from api.tasks.ingest import ingest_images_task

router = APIRouter()
log = structlog.get_logger()

IMAGES_INGESTED = Counter(
    "images_ingested_total",
    "Total images submitted for ingestion",
)


@router.post("", response_model=ImageIngestResponse, status_code=202)
async def ingest_images(request: ImageIngestRequest, db: DbSession):
    urls = [str(url) for url in request.urls]

    unique_urls = list(dict.fromkeys(urls))
    duplicates = len(urls) - len(unique_urls)

    job_id = f"ingest-{uuid.uuid4().hex[:12]}"

    ingest_images_task.apply_async(
        args=(unique_urls,),
        kwargs={
            "tags": request.tags,
            "priority": request.priority,
            "callback_url": str(request.callback_url) if request.callback_url else None,
        },
        task_id=job_id,
    )

    IMAGES_INGESTED.inc(len(unique_urls))

    log.info("images_queued", job_id=job_id, count=len(unique_urls), duplicates=duplicates)

    return ImageIngestResponse(
        job_id=job_id,
        queued=len(unique_urls),
        duplicates=duplicates,
        message=f"Queued {len(unique_urls)} images for processing",
    )


@router.get("", response_model=ImageListResponse)
async def list_images(
    db: DbSession,
    storage: StorageDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: ImageStatus | None = Query(default=None, description="Filter by status"),
    dataset: str | None = Query(default=None, description="Filter by dataset"),
    sort: Literal["newest", "oldest"] = Query(default="newest"),
) -> ImageListResponse:
    query = select(ORMImage)

    if dataset:
        query = query.where(ORMImage.dataset == dataset)

    if status:
        query = query.join(Processing, ORMImage.id == Processing.image_id, isouter=True)

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
        query = query.order_by(ORMImage.id.desc())
    else:
        query = query.order_by(ORMImage.id.desc())

    query = query.limit(limit).offset(offset)

    images = db.execute(query).scalars().all()

    results = []
    for img in images:
        proc = db.query(Processing).filter_by(image_id=img.id).first()
        ann = db.query(Annotation).filter_by(image_id=img.id).first()

        if proc:
            if proc.embed_status == "done":
                img_status = ImageStatus.DONE

            elif any(
                s == "failed" for s in [proc.ocr_status, proc.caption_status, proc.embed_status]
            ):
                img_status = ImageStatus.FAILED
            elif proc.embed_status == "running":
                img_status = ImageStatus.EMBEDDING
            elif proc.caption_status == "running" or proc.ocr_status == "running":
                img_status = ImageStatus.ANNOTATING
            else:
                img_status = ImageStatus.PENDING

        else:
            img_status = ImageStatus.PENDING

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
                ocr_status=proc.ocr_status if proc else None,
                caption_status=proc.caption_status if proc else None,
                embed_status=proc.embed_status if proc else None,
                caption=ann.caption_text if ann else None,
                ocr_text=ann.ocr_text if ann else None,
                tags=[],  # TODO: Parse tags from annotation
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
async def get_image(image_id: int, db: DbSession, storage: StorageDep) -> ImageResponse:
    img = db.query(ORMImage).filter_by(id=image_id).first()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    proc = db.query(Processing).filter_by(image_id=img.id).first()
    ann = db.query(Annotation).filter_by(image_id=img.id).first()

    if proc:
        if proc.embed_status == "done":
            img_status = ImageStatus.DONE

        elif any(s == "failed" for s in [proc.ocr_status, proc.caption_status, proc.embed_status]):
            img_status = ImageStatus.FAILED
        else:
            img_status = ImageStatus.PENDING

    else:
        img_status = ImageStatus.PENDING

    url = None
    if img.s3_key:
        url = storage.generate_presigned_url(
            img.s3_key, expiration=settings.s3_presigned_url_expiry
        )
    return ImageResponse(
        id=img.id,
        sha256=img.sha256,
        url=url,
        dataset=img.dataset,
        s3_key=img.s3_key,
        width=img.width,
        height=img.height,
        format=img.format,
        phash=img.phash,
        status=img_status,
        ocr_status=proc.ocr_status if proc else None,
        caption_status=proc.caption_status if proc else None,
        embed_status=proc.embed_status if proc else None,
        caption=ann.caption_text if ann else None,
        ocr_text=ann.ocr_text if ann else None,
        tags=[],
        created_at=img.created_at,
    )


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: int,
    db: DbSession,
    storage: StorageDep,
) -> None:
    img = db.query(ORMImage).filter_by(id=image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    if img.s3_key:
        try:
            storage.delete(img.s3_key)
        except Exception as e:
            log.error("s3_delete_failed", image_id=image_id, s3_key=img.s3_key, error=str(e))

    proc = db.query(Processing).filter_by(image_id=image_id).first()
    if proc and proc.embed_s3_key:
        try:
            storage.delete(proc.embed_s3_key)
            text_key = proc.embed_s3_key.replace(".npy", "_text.npy")
            storage.delete(text_key)
        except Exception as e:
            log.error(
                "s3_delete_failed",
                image_id=image_id,
                s3_key=proc.embed_s3_key,
                error=str(e),
            )

    db.query(Annotation).filter_by(image_id=image_id).delete()
    db.query(Processing).filter_by(image_id=image_id).delete()
    db.query(Artifact).filter_by(image_id=image_id).delete()
    db.delete(img)
    db.commit()

    log.info("image_deleted", image_id=image_id)
