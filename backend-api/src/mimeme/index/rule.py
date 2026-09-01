from __future__ import annotations

from datetime import datetime, timedelta

TASK_QUEUE = "mimeme-v2"
WORKFLOW = "mimeme.index.rebuild.v2"
PREPARE_ACTIVITY = "mimeme.index.prepare.v2"
SEAL_ACTIVITY = "mimeme.index.seal.v2"
BUILD_ACTIVITY = "mimeme.index.build.v2"
BGE_ACTIVITY = "mimeme.index.encode-bge.v1"
ACTIVATE_ACTIVITY = "mimeme.index.activate.v2"
SCHEDULE_ID = "search-index-rebuild-v2"
HEARTBEAT_TIMEOUT_S = 30
POLL_INTERVAL_S = 5.0
BUSY_WAIT_S = 30
BUSY_ATTEMPTS_PER_RUN = 20
PREPARE_MAX_ATTEMPTS = 5
SEAL_MAX_ATTEMPTS = 2
BUILD_MAX_ATTEMPTS = 3
BGE_MAX_ATTEMPTS = 3
ACTIVATE_MAX_ATTEMPTS = 5


def workflow_id(job_id: str) -> str:
    return f"rebuild-index-v2-{job_id}"


def settled(
    *,
    now: datetime,
    last_dirty_at: datetime | None,
    last_reconciled_at: datetime | None,
    settle: timedelta,
    max_stale: timedelta,
) -> bool:
    if last_dirty_at is None:
        return True
    if now - last_dirty_at >= settle:
        return True
    if last_reconciled_at is None:
        return True
    return now - last_reconciled_at >= max_stale
