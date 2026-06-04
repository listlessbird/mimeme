import time

import structlog
from shared.logging import emit_activity_event
from temporalio import activity

from activities.gpu_backends import get_gpu_backend
from activities.vision.models import AnnotateImageInput, AnnotateImageOutput


@activity.defn
async def annotate_image_activity(input: AnnotateImageInput) -> AnnotateImageOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None

    log = structlog.get_logger().bind(
        activity_name="annotate_image_activity",
        image_id=input.image_id,
        s3_key=input.s3_key,
    )

    try:
        backend = get_gpu_backend()
        log.info("activity_step", step="start", backend=type(backend).__name__)
        result = await backend.annotate_image(input)
        log.info(
            "activity_step",
            step="complete",
            caption_model=result.caption_model,
            ocr_model=result.ocr_model,
            caption_chars=len(result.caption),
            ocr_chars=len(result.ocr_text),
        )
        return result

    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        log.error("activity_step", step="failed", error=error_message, exc_info=True)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="annotate_image_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            image_id=input.image_id,
            s3_key=input.s3_key,
            length=input.length,
        )
