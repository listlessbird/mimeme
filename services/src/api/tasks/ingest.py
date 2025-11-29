import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from api.config import settings
from api.tasks import celery_app
from ingestion.annotate import _process_single_image
from ingestion.embeddings.base import EmbedderConfig
from ingestion.embeddings.db_ops import ensure_artifact_dir, get_image_text, save_embeddings
from ingestion.embeddings.pipeline import create_embedder
from ingestion.hashing import compute_phash, compute_sha256
from ingestion.imaging import image_info
from ingestion.orm import Image as ORMImage
from ingestion.orm import Processing, session_scope
from ingestion.repositories import images as images_repo
from ingestion.vision.base import create_vision_model

log = structlog.get_logger()


@celery_app.task(
    bind=True, name="api.tasks.ingest.ingest_images", max_retries=3, default_retry_delay=60
)
def ingest_images_task(
    self,
    urls: list[str],
    tags: list[str] | None = None,
    priority: str = "normal",
    callback_url: str | None = None,
) -> dict[str, Any]:
    total = len(urls)
    processed = 0
    failed = 0
    duplicates = 0
    result: list[dict] = []

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

            sha256 = compute_sha256(local_path)
            phash = compute_phash(local_path)

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
                    local_path.unlink(missing_ok=True)
                    continue

            width, height, fmt = image_info(local_path)

            rel_dir = Path(sha256[:2]) / sha256[2:4]
            rel_path = rel_dir / f"{sha256}.{fmt or 'jpg'}"

            dest_dir = settings.image_root / rel_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = settings.image_root / rel_path

            local_path.rename(dest_path)

            with session_scope() as session:
                images_repo.bulk_upsert(
                    session,
                    [
                        {
                            "sha256": sha256,
                            "rel_path": str(rel_path),
                            "width": width,
                            "height": height,
                            "format": fmt,
                            "phash": phash,
                        }
                    ],
                )
                session.commit()

                img = session.query(ORMImage).filter_by(sha256=sha256).first()
                if img is None:
                    raise RuntimeError(f"Failed to retrieve image after insert: {sha256}")
                image_id = img.id

                proc = Processing(image_id=image_id)
                session.add(proc)
                session.commit()

                if vision_model is None:
                    log.info("loading_vision_model", model=settings.vision_model)

                    vision_model = create_vision_model(settings.vision_model)

                with session_scope() as session:
                    img = session.query(ORMImage).filter_by(id=image_id).first()
                    if img is None:
                        raise RuntimeError(f"Failed to retrieve image for annotation: {image_id}")
                    annotation_result = _process_single_image(session, vision_model, img)

                if embedder is None:
                    log.info("loading_embedder")
                    cfg = EmbedderConfig(
                        image_model=settings.embed_model, device=settings.embed_device
                    )
                    embedder = create_embedder(cfg)

                text = get_image_text(image_id)

                artifact_dir = ensure_artifact_dir(embedder.image_model_name)

                batch_results = embedder.embed_batch(
                    image_paths=[(image_id, str(dest_path))], text_provider=lambda _: text
                )

                if batch_results:
                    img_id, img_emb, txt_emb = batch_results[0]
                    save_embeddings(
                        image_id=img_id,
                        image_embedding=img_emb,
                        text_embedding=txt_emb,
                        artifact_dir=artifact_dir,
                        model_name=embedder.image_model_name,
                    )
                processed += 1
                result.append(
                    {
                        "url": url,
                        "status": "success",
                        "image_id": image_id,
                        "sha256": sha256,
                    }
                )

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
