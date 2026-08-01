from __future__ import annotations

TASK_QUEUE = "mimeme-v2"
WORKFLOW = "mimeme.index.rebuild.v2"
PREPARE_ACTIVITY = "mimeme.index.prepare.v2"
BUILD_ACTIVITY = "mimeme.index.build.v2"
ACTIVATE_ACTIVITY = "mimeme.index.activate.v2"
SCHEDULE_ID = "search-index-rebuild-v2"
HEARTBEAT_TIMEOUT_S = 30
POLL_INTERVAL_S = 5.0
BUSY_WAIT_S = 30
BUSY_ATTEMPTS_PER_RUN = 20
PREPARE_MAX_ATTEMPTS = 5
BUILD_MAX_ATTEMPTS = 3
ACTIVATE_MAX_ATTEMPTS = 5


def workflow_id(job_id: str) -> str:
    return f"rebuild-index-v2-{job_id}"
