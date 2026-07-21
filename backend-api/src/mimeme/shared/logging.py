from __future__ import annotations

import atexit
import logging
import time
from typing import cast

import structlog
from axiom_py import Client
from axiom_py.structlog import AxiomProcessor
from temporalio import activity

from mimeme.shared.config import settings

_axiom_client: Client | None = None


def _get_axiom_processor() -> AxiomProcessor | None:
    if (
        not settings.logging.axiom_api_token.get_secret_value()
        or not settings.logging.axiom_dataset
    ):
        return None

    global _axiom_client

    _axiom_client = Client(token=settings.logging.axiom_api_token.get_secret_value())
    atexit.register(_axiom_client.shutdown_hook)
    return AxiomProcessor(_axiom_client, settings.logging.axiom_dataset)


def setup_logging(service: str = "api") -> None:
    # Keep SQL statement logs off by default; they are too noisy for worker tracing.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    axiom_proc = _get_axiom_processor()

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", key="_time"),
    ]

    if axiom_proc:
        processors.append(cast(structlog.types.Processor, axiom_proc))

    processors.append(
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.logging.level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        service=service,
        app_env=settings.app_env,
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
        "activity_id": info.activity_id,
        "activity_type": info.activity_type,
        "attempt": info.attempt,
        "task_queue": info.task_queue,
        "is_local": info.is_local,
    }


def emit_activity_event(
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
        **activity_context(),
    }
    event.update(fields)
    if error:
        event["error"] = error
    log.info("activity_wide_event", **event)
