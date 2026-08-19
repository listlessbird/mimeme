from __future__ import annotations

import logging
import time

import structlog
from temporalio import activity

from mimeme import release
from mimeme.config import Settings


def setup_logging(settings: Settings, service: str = "api") -> None:
    # Keep SQL statement logs off by default; they are too noisy for worker tracing.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", key="_time"),
    ]

    processors.append(
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(settings.logging.level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        service=service,
        app_env=settings.app_env,
        release_id=release.ID,
    )


def activity_context() -> dict[str, object]:
    try:
        info = activity.info()
    except RuntimeError:
        return {}
    return {
        "workflow_id": info.workflow_id,
        "run_id": info.workflow_run_id,
        "workflow_type": info.workflow_type,
        "workflow_name": info.workflow_type,
        "activity_id": info.activity_id,
        "activity_type": info.activity_type,
        "attempt": info.attempt,
        "task_queue": info.task_queue,
        "is_local": info.is_local,
    }


def emit_activity_event(
    *,
    log: structlog.BoundLogger,
    event_name: str,
    activity_name: str,
    started_at: float,
    outcome: str,
    error: str | None = None,
    **fields: object,
) -> None:
    """Emit one canonical, context-rich event for a Temporal activity attempt."""
    event: dict[str, object] = {
        "activity_name": activity_name,
        "outcome": outcome,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        **activity_context(),
    }
    event.update(fields)
    if event["duration_ms"] is None:
        event["duration_ms"] = int((time.monotonic() - started_at) * 1000)
    if error:
        event["error"] = error
    writer = log.error if outcome in {"failed", "error"} else log.info
    writer(event_name, **event)
