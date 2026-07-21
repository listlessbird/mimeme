from __future__ import annotations

import time

import structlog
from temporalio import activity

from mimeme.activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
)
from mimeme.activities.gpu_backends import get_gpu_backend
from mimeme.shared.logging import emit_activity_event


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
        emit_activity_event(
            log=log,
            activity_name="embed_batch_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            item_count=len(input.items),
            dataset=input.dataset,
        )
