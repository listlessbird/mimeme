from __future__ import annotations

import time

import structlog
from temporalio import activity

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EncodeQueryInput,
    EncodeQueryOutput,
)
from activities.gpu_backends import get_gpu_backend


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
async def embed_batch_activity(input: EmbedBatchInput) -> EmbedBatchOutput:
    started = time.monotonic()
    log = structlog.get_logger().bind(
        activity_name="embed_batch_activity",
        item_count=len(input.items),
        dataset=input.dataset,
    )
    outcome = "success"
    error_message: str | None = None

    try:
        backend = get_gpu_backend()

        log.info("activity_step", step="start", backend=type(backend).__name__)

        result = await backend.embed_batch(input)

        if result.failed_ids:
            outcome = "partial_failure"

        log.info(
            "activity_step",
            step="complete",
            processed=len(result.results),
            failed=len(result.failed_ids),
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
            activity_name="embed_batch_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            item_count=len(input.items),
            dataset=input.dataset,
        )


@activity.defn
async def encode_query_activity(input: EncodeQueryInput) -> EncodeQueryOutput:
    started = time.monotonic()
    log = structlog.get_logger().bind(activity_name="encode_query_activity")
    outcome = "success"
    error_message: str | None = None
    try:
        backend = get_gpu_backend()
        log.info("activity_step", step="start", backend=type(backend).__name__)
        result = await backend.encode_query(input)
        return result
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        log.error("activity_step", step="failed", error=error_message, exc_info=True)
        raise
    finally:
        _emit_activity_event(
            log=log,
            activity_name="encode_query_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            query_chars=len(input.query),
        )
