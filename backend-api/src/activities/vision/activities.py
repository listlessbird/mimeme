import time

import structlog
from temporalio import activity

from activities.gpu_backends import get_gpu_backend
from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput


def _activity_context() -> dict[str, object]:
    try:
        info = activity.info()
    except RuntimeError:
        return {}
    return {
        "workflow_id": info.workflow_id,
        "run_id": info.workflow_run_id,
        "workflow_type": info.workflow_type,
        "activity_id": info.activity_id,
        "activity_type": info.activity_type,
        "attempt": info.attempt,
        "task_queue": info.task_queue,
        "is_local": info.is_local,
    }


def _emit_activity_event(
    *,
    log: structlog.BoundLogger,
    activity_name: str,
    started_at: float,
    outcome: str,
    error: str | None = None,
    **fields: object,
) -> None:
    event: dict[str, object] = {
        "event_type": "activity_wide_event",
        "activity_name": activity_name,
        "outcome": outcome,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        **_activity_context(),
    }
    event.update(fields)
    if error:
        event["error"] = error
    log.info("activity_wide_event", **event)


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
        _emit_activity_event(
            log=log,
            activity_name="caption_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            image_id=input.image_id,
            s3_key=input.s3_key,
            length=input.length,
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
        _emit_activity_event(
            log=log,
            activity_name="ocr_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            image_id=input.image_id,
            s3_key=input.s3_key,
        )
