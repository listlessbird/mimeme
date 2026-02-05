import tempfile
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx
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


@activity.defn
async def download_image_activity(input: DownloadImageInput) -> DownloadImageOutput:
    try:
        parsed = urlparse(input.url)
        filename = Path(parsed.path).name or "image"

        with httpx.Client(timeout=30.0, follow_redirects=True) as httpclient:
            response = httpclient.get(input.url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if not content_type.startswith("image/"):
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

                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path=f.name,
                    filename=filename,
                    success=True,
                )

    except Exception as e:
        return DownloadImageOutput(
            ingest_url_id=input.ingest_url_id,
            local_path="",
            filename="",
            success=False,
            error=str(e),
        )


@activity.defn
async def process_image_activity(input: ProcessImageInput) -> ProcessImageOutput:
    storage = cast(StorageService, get_storage_service())

    local_path = Path(input.local_path)

    try:
        sha256 = compute_sha256(local_path)
        with session_scope() as session:
            existing = session.query(ORMImage).filter_by(sha256=sha256).first()

            if existing:
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
        local_path.unlink(missing_ok=True)


@activity.defn
async def cleanup_temp_file_activity(local_path: str) -> None:
    Path(local_path).unlink(missing_ok=True)
