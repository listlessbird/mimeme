import tempfile
import time
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session
from temporalio import activity

from mimeme.activities.storage.img_utils import compute_phash, compute_sha256, get_image_info
from mimeme.activities.storage.models import (
    CleanupStagedUploadInput,
    DownloadImageInput,
    DownloadImageOutput,
    ProcessImageInput,
    ProcessImageOutput,
)
from mimeme.activities.storage.phash_index import get_phash_index
from mimeme.db.schema import DuplicateReason, ORMImage, Processing, ProcessingStatus
from mimeme.domain.image_ingest_input import RemoteImageUrlInput, StagedUploadInput
from mimeme.shared.db import session_scope
from mimeme.shared.logging import emit_activity_event
from mimeme.shared.services import get_artifact_storage_service, get_media_storage_service

log = structlog.get_logger()

IMAGE_ACCEPT_HEADER = "image/avif,image/webp,image/apng,image/png,image/jpeg,image/*,*/*;q=0.8"


class DuplicateProcessingFields(TypedDict):
    needs_annotation: bool
    needs_embedding: bool
    existing_caption: str | None
    existing_ocr_text: str | None


def _duplicate_processing_fields(session: Session, image: ORMImage) -> DuplicateProcessingFields:
    proc = image.processing
    if proc is None:
        proc = Processing(image_id=image.id)
        image.processing = proc
        session.add(proc)

    annotation = image.annotation
    needs_annotation = (
        annotation is None
        or proc.caption_status != ProcessingStatus.DONE
        or proc.ocr_status != ProcessingStatus.DONE
    )
    needs_embedding = proc.embed_status != ProcessingStatus.DONE or not proc.embed_s3_key

    return {
        "needs_annotation": needs_annotation,
        "needs_embedding": needs_embedding,
        "existing_caption": annotation.caption_text if annotation else None,
        "existing_ocr_text": annotation.ocr_text if annotation else None,
    }


@activity.defn
def download_image_activity(input: DownloadImageInput) -> DownloadImageOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    bytes_downloaded: int | None = None
    source_url = input.input.url if isinstance(input.input, RemoteImageUrlInput) else None
    log.info(
        "activity_step",
        activity_name="download_image_activity",
        step="start_download",
        job_id=input.job_id,
        ingest_url_id=input.ingest_url_id,
        input_kind=input.input.kind,
    )
    try:
        if isinstance(input.input, StagedUploadInput):
            filename = Path(input.input.artifact_key).name or "image"
            content = get_artifact_storage_service().download_bytes(input.input.artifact_key)
            suffix = Path(filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(content)
                bytes_downloaded = len(content)
                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path=f.name,
                    filename=filename,
                    success=True,
                )

        parsed = urlparse(input.input.url)
        filename = Path(parsed.path).name or "image"

        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "user-agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
                ),
                "accept": IMAGE_ACCEPT_HEADER,
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
        ) as httpclient:
            response = httpclient.get(input.input.url)
            log.info(
                "activity_step",
                activity_name="download_image_activity",
                step="response_received",
                job_id=input.job_id,
                ingest_url_id=input.ingest_url_id,
                status_code=response.status_code,
            )
            status_code = response.status_code
            content_type = response.headers.get("content-type")
            response.raise_for_status()

            resolved_content_type = response.headers.get("content-type", "")

            if not resolved_content_type.startswith("image/"):
                outcome = "failed"
                error_message = f"non_image_content_type:{resolved_content_type}"
                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path="",
                    filename="",
                    success=False,
                    error=f"Couldnt resolve an image from the url {input.input.url}, got {resolved_content_type} as content type",
                )

            suffix = Path(filename).suffix or ".jpg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                bytes_downloaded = len(response.content)

                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path=f.name,
                    filename=filename,
                    success=True,
                )

    except httpx.HTTPStatusError as e:
        if e.response is not None:
            status_code = e.response.status_code
            content_type = e.response.headers.get("content-type")

        if (
            status_code is not None
            and 400 <= status_code < 500
            and status_code
            not in {
                408,
                425,
                429,
            }
        ):
            log.info(
                "activity_step",
                activity_name="download_image_activity",
                step="terminal_http_error",
                job_id=input.job_id,
                ingest_url_id=input.ingest_url_id,
                status_code=status_code,
                error=str(e),
            )
            outcome = "failed"
            error_message = str(e)
            return DownloadImageOutput(
                ingest_url_id=input.ingest_url_id,
                local_path="",
                filename="",
                success=False,
                error=str(e),
            )

        log.info(
            "activity_step",
            activity_name="download_image_activity",
            step="retryable_http_error",
            job_id=input.job_id,
            ingest_url_id=input.ingest_url_id,
            status_code=status_code,
            error=str(e),
        )
        outcome = "error"
        error_message = str(e)
        raise
    except httpx.TransportError as e:
        log.info(
            "activity_step",
            activity_name="download_image_activity",
            step="retryable_transport_error",
            job_id=input.job_id,
            ingest_url_id=input.ingest_url_id,
            error=str(e),
        )
        outcome = "error"
        error_message = str(e)
        raise
    except Exception as e:
        log.info(
            "activity_step",
            activity_name="download_image_activity",
            step="unexpected_download_error",
            job_id=input.job_id,
            ingest_url_id=input.ingest_url_id,
            error=str(e),
        )
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="download_image_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            job_id=input.job_id,
            ingest_url_id=input.ingest_url_id,
            url=source_url,
            input_kind=input.input.kind,
            status_code=status_code,
            content_type=content_type,
            bytes=bytes_downloaded,
        )


@activity.defn
def cleanup_staged_upload_activity(input: CleanupStagedUploadInput) -> None:
    try:
        get_artifact_storage_service().delete(input.artifact_key)
    except Exception:
        log.warning(
            "staged_upload_cleanup_failed",
            artifact_key=input.artifact_key,
            exc_info=True,
        )


@activity.defn
def process_image_activity(input: ProcessImageInput) -> ProcessImageOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    image_id: int | None = None
    is_duplicate: bool | None = None
    s3_key: str | None = None
    storage = get_media_storage_service()

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
            existing = session.scalars(select(ORMImage).where(ORMImage.sha256 == sha256)).first()

            if existing:
                duplicate_processing = _duplicate_processing_fields(session, existing)
                image_id = existing.id
                is_duplicate = True
                s3_key = existing.s3_key or ""
                outcome = "duplicate"
                log.info(
                    "activity_step",
                    activity_name="process_image_activity",
                    step="duplicate_detected",
                    ingest_url_id=input.ingest_url_id,
                    image_id=existing.id,
                    duplicate_reason=DuplicateReason.SHA256,
                    duplicate_of_image_id=existing.id,
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
                    duplicate_reason=DuplicateReason.SHA256,
                    duplicate_of_image_id=existing.id,
                    **duplicate_processing,
                )

        phash = compute_phash(local_path)

        matched_image_id = get_phash_index().match(phash)

        if matched_image_id is not None:
            with session_scope() as session:
                existing = session.get(ORMImage, matched_image_id)
                if existing is None:
                    raise ValueError(f"matched duplicate image {matched_image_id} not found")
                duplicate_processing = _duplicate_processing_fields(session, existing)
                canonical_sha256 = existing.sha256
                canonical_s3_key = existing.s3_key or ""
                width = existing.width
                height = existing.height
                format = existing.format

            return ProcessImageOutput(
                ingest_url_id=input.ingest_url_id,
                image_id=matched_image_id,
                sha256=canonical_sha256,
                s3_key=canonical_s3_key,
                width=width,
                height=height,
                format=format,
                is_duplicate=True,
                duplicate_reason=DuplicateReason.PHASH,
                duplicate_of_image_id=matched_image_id,
                **duplicate_processing,
            )

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

            get_phash_index().add(image_id, phash)

            is_duplicate = False

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
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise

    finally:
        log.info(
            "activity_step",
            activity_name="process_image_activity",
            step="cleanup_temp_file",
            ingest_url_id=input.ingest_url_id,
            local_path=str(local_path),
        )
        local_path.unlink(missing_ok=True)
        emit_activity_event(
            log=log,
            activity_name="process_image_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            ingest_url_id=input.ingest_url_id,
            dataset=input.dataset,
            local_path=str(local_path),
            image_id=image_id,
            is_duplicate=is_duplicate,
            s3_key=s3_key,
        )


@activity.defn
def cleanup_temp_file_activity(local_path: str) -> None:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    try:
        Path(local_path).unlink(missing_ok=True)
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="cleanup_temp_file_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            local_path=local_path,
        )
