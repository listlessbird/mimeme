import tempfile
import time
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx
import structlog
from temporalio import activity

from activities.storage.img_utils import compute_phash, compute_sha256, get_image_info
from activities.storage.models import (
    DownloadImageInput,
    DownloadImageOutput,
    ProcessImageInput,
    ProcessImageOutput,
)
from shared.db import session_scope
from shared.models import ORMImage, Processing
from shared.services import StorageService, get_storage_service

log = structlog.get_logger()


@activity.defn
async def download_image_activity(input: DownloadImageInput) -> DownloadImageOutput:
    started = time.monotonic()
    event: dict[str, object] = {
        "event_type": "activity_wide_event",
        "activity_name": "download_image_activity",
        "job_id": input.job_id,
        "ingest_url_id": input.ingest_url_id,
        "url": input.url,
    }
    log.info(
        "activity_step",
        activity_name="download_image_activity",
        step="start_download",
        job_id=input.job_id,
        ingest_url_id=input.ingest_url_id,
        url=input.url,
    )
    try:
        parsed = urlparse(input.url)
        filename = Path(parsed.path).name or "image"

        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "user-agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
                ),
                "accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
        ) as httpclient:
            response = httpclient.get(input.url)
            log.info(
                "activity_step",
                activity_name="download_image_activity",
                step="response_received",
                job_id=input.job_id,
                ingest_url_id=input.ingest_url_id,
                status_code=response.status_code,
            )
            event["status_code"] = response.status_code
            event["content_type"] = response.headers.get("content-type")
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if not content_type.startswith("image/"):
                event["outcome"] = "failed"
                event["error"] = f"non_image_content_type:{content_type}"
                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path="",
                    filename="",
                    success=False,
                    error=f"Couldnt resolve an image from the url {input.url}, got {content_type} as content type",
                )

            suffix = Path(filename).suffix or ".jpg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                event["outcome"] = "success"
                event["bytes"] = len(response.content)

                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path=f.name,
                    filename=filename,
                    success=True,
                )

    except Exception as e:
        log.info(
            "activity_step",
            activity_name="download_image_activity",
            step="download_error",
            job_id=input.job_id,
            ingest_url_id=input.ingest_url_id,
            error=str(e),
        )
        event["outcome"] = "failed"
        event["error"] = str(e)
        return DownloadImageOutput(
            ingest_url_id=input.ingest_url_id,
            local_path="",
            filename="",
            success=False,
            error=str(e),
        )
    finally:
        event["duration_ms"] = int((time.monotonic() - started) * 1000)
        log.info("activity_wide_event", **event)


@activity.defn
async def process_image_activity(input: ProcessImageInput) -> ProcessImageOutput:
    storage = cast(StorageService, get_storage_service())

    local_path = Path(input.local_path)
    log.info(
        "activity_step",
        activity_name="process_image_activity",
        step="start_process",
        ingest_url_id=input.ingest_url_id,
        local_path=str(local_path),
        dataset=input.dataset,
    )

    try:
        sha256 = compute_sha256(local_path)
        with session_scope() as session:
            existing = session.query(ORMImage).filter_by(sha256=sha256).first()

            if existing:
                log.info(
                    "activity_step",
                    activity_name="process_image_activity",
                    step="duplicate_detected",
                    ingest_url_id=input.ingest_url_id,
                    image_id=existing.id,
                )
                return ProcessImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    image_id=existing.id,
                    sha256=sha256,
                    s3_key=existing.s3_key or "",
                    width=existing.width,
                    height=existing.height,
                    format=existing.format,
                    is_duplicate=True,
                )

        phash = compute_phash(local_path)
        width, height, format = get_image_info(local_path)

        file_size = local_path.stat().st_size

        ext = format or "jpg"
        s3_key = storage.build_image_key(sha256, input.dataset, ext)
        etag = storage.upload_file(local_path, s3_key)

        with session_scope() as session:
            img = ORMImage(
                sha256=sha256,
                dataset=input.dataset,
                original_filename=input.filename,
                s3_key=s3_key,
                s3_etag=etag,
                width=width,
                height=height,
                format=format,
                file_size=file_size,
                phash=phash,
            )

            session.add(img)
            session.flush()
            image_id = img.id

            proc = Processing(image_id=image_id)
            session.add(proc)
        log.info(
            "activity_step",
            activity_name="process_image_activity",
            step="image_processed",
            ingest_url_id=input.ingest_url_id,
            image_id=image_id,
            s3_key=s3_key,
        )

        return ProcessImageOutput(
            ingest_url_id=input.ingest_url_id,
            image_id=image_id,
            sha256=sha256,
            s3_key=s3_key,
            width=width,
            height=height,
            format=format,
            is_duplicate=False,
        )

    finally:
        log.info(
            "activity_step",
            activity_name="process_image_activity",
            step="cleanup_temp_file",
            ingest_url_id=input.ingest_url_id,
            local_path=str(local_path),
        )
        local_path.unlink(missing_ok=True)


@activity.defn
async def cleanup_temp_file_activity(local_path: str) -> None:
    Path(local_path).unlink(missing_ok=True)
