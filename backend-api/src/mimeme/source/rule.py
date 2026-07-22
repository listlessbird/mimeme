from __future__ import annotations

TASK_QUEUE = "mimeme-v2"

SYNC_WORKFLOW = "mimeme.source.sync.v2"
RETRY_WORKFLOW = "mimeme.source.retry.v2"
DISCOVER_ACTIVITY = "mimeme.source.discover.v2"
FINISH_ACTIVITY = "mimeme.source.finish.v2"

SCHEDULE_PREFIX = "source-sync-v2-"

HEARTBEAT_TIMEOUT_S = 30.0

ERROR_LIMIT = 1000

_DEFAULT_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "accept": "application/json",
}

RETRYABLE_4XX = frozenset({408, 425, 429})


def schedule_id(source_id: int) -> str:
    return f"{SCHEDULE_PREFIX}{source_id}"


def scheduled_workflow_id(source_id: int) -> str:
    return f"{SCHEDULE_PREFIX}{source_id}"


def manual_workflow_id(source_id: int, request_id: str) -> str:
    return f"{SCHEDULE_PREFIX}{source_id}-{request_id}"


def retry_workflow_id(source_run_id: int, request_id: str) -> str:
    return f"source-retry-v2-{source_run_id}-{request_id}"


def is_terminal_http_status(status: int) -> bool:
    return 400 <= status < 500 and status not in RETRYABLE_4XX


def truncate_error(message: str, limit: int = ERROR_LIMIT) -> str:
    cleaned = message.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."
