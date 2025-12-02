import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from api.config import settings
from api.models.orm import Annotation, Processing, Session, session_scope
from api.models.orm import Image as ORMImage
from api.services.storage import get_storage_service
from api.tasks import celery_app
from ingestion.annotate import now_iso
from ingestion.hashing import compute_phash, compute_sha256
from ingestion.imaging import image_info
from ingestion.vision.base import create_vision_model

log = structlog.get_logger()


@celery_app.task(
    bind=True, name="api.tasks.ingest.ingest_images", max_retries=3, default_retry_delay=60
)
def ingest_images_task(
    self,
    urls: list[str],
    tags: list[str] | None = None,
    dataset: str | None = None,
    priority: str = "normal",
    callback_url: str | None = None,
) -> dict[str, Any]:
    total = len(urls)
    processed = 0
    failed = 0
    duplicates = 0
    result: list[dict] = []

    storage = get_storage_service()
    storage.ensure_bucket_exists()

    self.update_state(
        state="PROGRESS",
        meta={"progress": 0, "message": f"Starting ingestion of {total} images"},
    )

    vision_model = None
    embedder = None

    for i, url in enumerate(urls):
        try:
            progress = (i / total) * 100

            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "message": f"Processing {i + 1}/{total}: {url[:50]}...",
                    "current_url": url,
                },
            )

            local_path, filename = _download_image(url)

            if local_path is None:
                failed += 1
                result.append({"url": url, "status": "download_failed"})
                continue

            try:
                sha256 = compute_sha256(local_path)
                phash = compute_phash(local_path)
                width, height, fmt = image_info(local_path)
                file_size = local_path.stat().st_size

                with session_scope() as session:
                    existing = session.query(ORMImage).filter_by(sha256=sha256).first()

                    if existing:
                        duplicates += 1
                        result.append(
                            {
                                "url": url,
                                "status": "duplicate",
                                "existing_id": existing.id,
                            }
                        )
                        continue
                ext = fmt or "jpg"
                s3_key = storage.build_image_key(sha256, dataset, ext)
                etag = storage.upload_file(local_path, s3_key)

                with session_scope() as session:
                    img = ORMImage(
                        sha256=sha256,
                        dataset=dataset,
                        original_filename=filename,
                        s3_key=s3_key,
                        s3_etag=etag,
                        width=width,
                        height=height,
                        format=fmt,
                        file_size=file_size,
                        phash=phash,
                    )
                    session.add(img)
                    session.flush()
                    image_id = img.id

                    proc = Processing(image_id=image_id)
                    session.add(proc)
                    session.commit()

                if vision_model is None:
                    log.info("loading_vision_model", model=settings.vision_model)
                    vision_model = create_vision_model(settings.vision_model)

                with session_scope() as session:
                    img = session.query(ORMImage).filter_by(id=image_id).first()
                    if img:
                        _annotate_image(session, vision_model, img, local_path)

            finally:
                local_path.unlink(missing_ok=True)

        except Exception as e:
            log.exception("ingest_failed", url=url)
            failed += 1
            result.append({"url": url, "status": "failed", "error": str(e)})

    summary = {
        "total": total,
        "processed": processed,
        "failed": failed,
        "duplicates": duplicates,
        "results": result,
    }

    if callback_url:
        _call_webhook(callback_url, summary)

    log.info(
        "ingest_complete",
        total=total,
        processed=processed,
        failed=failed,
        duplicates=duplicates,
    )

    return summary


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _download_image(url: str) -> tuple[Path | None, str]:
    try:
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "image"

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                log.warning("not_an_img_probably", url=url, content_type=content_type)
                return None, ""

            suffix = Path(filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                return Path(f.name), filename

    except Exception as e:
        log.error("image_download_failed", url=url, error=str(e))
        return None, ""


def _call_webhook(url: str, data: dict) -> None:
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(url, json=data)
    except Exception as e:
        log.error("webhook_failed", url=url, error=str(e))


def _annotate_image(session: Session, vision_model, img: ORMImage, local_path: Path) -> None:
    try:
        pil = Image.open(local_path).convert("RGB")

        proc = session.query(Processing).filter_by(image_id=img.id).first()

        if not proc:
            proc = Processing(image_id=img.id)
            session.add(proc)
            session.flush()

        cap = vision_model.caption(pil, length="normal")
        ocr = vision_model.ocr(pil)

        ann = session.query(Annotation).filter_by(image_id=img.id).first()
        if not ann:
            ann = Annotation(image_id=img.id)
            session.add(ann)

        ann.caption_text = cap.caption
        ann.ocr_text = ocr.text

        proc.caption_status = "done"
        proc.caption_model = cap.model
        proc.caption_updated_at = now_iso()
        proc.ocr_model = ocr.model
        proc.ocr_updated_at = now_iso()

        session.commit()
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
