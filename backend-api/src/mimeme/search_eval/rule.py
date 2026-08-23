from __future__ import annotations

RUN_WORKFLOW = "mimeme.search_eval.run.v1"
SCORE_WORKFLOW = "mimeme.search_eval.score.v1"
PREPARE_ACTIVITY = "mimeme.search_eval.prepare.v1"
RETRIEVE_ACTIVITY = "mimeme.search_eval.retrieve.v1"
SCORE_ACTIVITY = "mimeme.search_eval.calculate.v1"
FAIL_ACTIVITY = "mimeme.search_eval.fail.v1"

BATCH_SIZE = 25
MAX_ATTEMPTS = 3
HEARTBEAT_TIMEOUT_S = 120


def run_workflow_id(run_id: str) -> str:
    return f"search-eval-v1-{run_id}"


def score_workflow_id(run_id: str, request_id: str) -> str:
    return f"search-eval-score-v1-{run_id}-{request_id}"
