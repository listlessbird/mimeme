import time

import structlog
from temporalio import activity

from activities.gpu_backends import get_gpu_backend
from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput


@activity.defn
async def caption_activity(input: CaptionInput) -> CaptionOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None

    log = structlog.get_logger().bind(
        activity_name="caption_activity",
        image_id=input.image_id,
        s3_key=input.s3_key,
    )

    try:
        backend = get_gpu_backend()
        log.info("activity_step", step="start", backend=type(backend).__name__)
        result = await backend.caption(input)
        log.info(
            "activity_step", step="complete", model=result.model, caption_chars=len(result.caption)
        )
        return result

    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        log.error("activity_step", step="failed", error=error_message, exc_info=True)
        raise
    finally:
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=error_message,
        )


@activity.defn
async def ocr_activity(input: OCRInput) -> OCROutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    log = structlog.get_logger().bind(
        activity_name="ocr_activity",
        image_id=input.image_id,
        s3_key=input.s3_key,
    )

    try:
        backend = get_gpu_backend()
        log.info("activity_step", step="start", backend=type(backend).__name__)
        result = await backend.ocr(input)
        log.info("activity_step", step="complete", model=result.model, text_chars=len(result.text))
        return result
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        log.error("activity_step", step="failed", error=error_message, exc_info=True)
        raise
    finally:
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=error_message,
        )
