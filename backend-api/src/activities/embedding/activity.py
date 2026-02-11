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


@activity.defn
async def embed_batch_activity(input: EmbedBatchInput) -> EmbedBatchOutput:
    started = time.monotonic()
    log = structlog.get_logger().bind(
        activity_name="embed_batch_activity",
        item_count=len(input.items),
        dataset=input.dataset,
    )
    outcome = "success"
    error_type: str | None = None
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
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=error_message,
        )


@activity.defn
async def encode_query_activity(input: EncodeQueryInput) -> EncodeQueryOutput:
    log = structlog.get_logger().bind(activity_name="encode_query_activity")
    backend = get_gpu_backend()
    log.info("activity_step", step="start", backend=type(backend).__name__)
    result = await backend.encode_query(input)
    log.info(
        "activity_wide_event",
        event_type="activity_wide_event",
        outcome="success",
        dimension=result.dimension,
    )

    return result
