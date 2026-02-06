import tempfile
import time
from pathlib import Path
from typing import cast

import structlog
from PIL import Image
from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput
from activities.vision.moondream import Moondream2
from shared.services import StorageService, get_storage_service


@activity.defn
async def caption_activity(input: CaptionInput) -> CaptionOutput:
    started = time.monotonic()
    outcome = "success"
    error_type: str | None = None
    error_message: str | None = None
    caption_chars = 0
    model_version: str | None = None
    log = structlog.get_logger().bind(
        activity_name="caption_activity",
        image_id=input.image_id,
        s3_key=input.s3_key,
        caption_length=input.length,
    )
    storage = cast(StorageService, get_storage_service())
    log.info("activity_step", step="start")

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            log.info("activity_step", step="download_start")
            storage.download_file(input.s3_key, tmp_path)
            log.info("activity_step", step="download_complete", tmp_path=str(tmp_path))

            pil_image = Image.open(tmp_path).convert("RGB")
            log.info("activity_step", step="image_loaded", width=pil_image.width, height=pil_image.height)

            try:
                model = Moondream2.get_instance()
            except (ImportError, OSError) as exc:
                raise ApplicationError(
                    (
                        "Vision model dependencies are missing or incompatible. "
                        "Run `python scripts/check_remote_model_deps.py --model-id vikhyatk/moondream2` "
                        "and install reported packages, then verify system image libs."
                    ),
                    non_retryable=True,
                ) from exc
            log.info("activity_step", step="model_ready")
            result = model.caption(pil_image, length=input.length)
            model_version = result.model
            caption_chars = len(result.caption)
            log.info("activity_step", step="caption_complete", model=model_version, caption_chars=caption_chars)

            return CaptionOutput(image_id=input.image_id, caption=result.caption, model=result.model)
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        log.error(
            "activity_step",
            step="failed",
            error_type=error_type,
            error=error_message,
            exc_info=True,
        )
        raise
    finally:
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            model=model_version,
            caption_chars=caption_chars,
            error_type=error_type,
            error=error_message,
        )


@activity.defn
async def ocr_activity(input: OCRInput) -> OCROutput:
    started = time.monotonic()
    outcome = "success"
    error_type: str | None = None
    error_message: str | None = None
    text_chars = 0
    model_version: str | None = None
    log = structlog.get_logger().bind(
        activity_name="ocr_activity",
        image_id=input.image_id,
        s3_key=input.s3_key,
    )
    storage = cast(StorageService, get_storage_service())
    log.info("activity_step", step="start")

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            log.info("activity_step", step="download_start")
            storage.download_file(input.s3_key, tmp_path)
            log.info("activity_step", step="download_complete", tmp_path=str(tmp_path))

            pil_image = Image.open(tmp_path).convert("RGB")
            log.info("activity_step", step="image_loaded", width=pil_image.width, height=pil_image.height)

            try:
                model = Moondream2.get_instance()
            except (ImportError, OSError) as exc:
                raise ApplicationError(
                    (
                        "Vision model dependencies are missing or incompatible. "
                        "Run `python scripts/check_remote_model_deps.py --model-id vikhyatk/moondream2` "
                        "and install reported packages, then verify system image libs."
                    ),
                    non_retryable=True,
                ) from exc
            log.info("activity_step", step="model_ready")
            result = model.ocr(pil_image)
            model_version = result.model
            text_chars = len(result.text)
            log.info("activity_step", step="ocr_complete", model=model_version, text_chars=text_chars)

            return OCROutput(
                image_id=input.image_id,
                text=result.text,
                model=result.model,
            )
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        log.error(
            "activity_step",
            step="failed",
            error_type=error_type,
            error=error_message,
            exc_info=True,
        )
        raise
    finally:
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            model=model_version,
            text_chars=text_chars,
            error_type=error_type,
            error=error_message,
        )
