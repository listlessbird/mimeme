from __future__ import annotations

from datetime import UTC, datetime, timedelta

from temporalio import activity

from mimeme.db import Db
from mimeme.job import ops
from mimeme.job.model import (
    CompleteRebuildCommand,
    FailRebuildCommand,
    PrepareCommand,
    RebuildStateCommand,
    RebuildStateOutput,
    ReconcileCommand,
    ReleaseCommand,
    StartCommand,
)

REBUILD_STATE = "mimeme.job.rebuild-state.tmp"


class JobActivities:
    def __init__(self, db: Db, *, rebuild_claim_timeout: timedelta) -> None:
        self._db = db
        self._claim_timeout = rebuild_claim_timeout

    @activity.defn(name=REBUILD_STATE)
    async def rebuild_state(self, command: RebuildStateCommand) -> RebuildStateOutput:
        match command:
            case PrepareCommand():
                decision = await ops.prepare_rebuild(
                    self._db,
                    job_id=command.job_id,
                    workflow_id=command.workflow_id,
                    force=command.force,
                    trigger=command.trigger,
                    now=datetime.now(UTC),
                    claim_timeout=self._claim_timeout,
                )
                return RebuildStateOutput(decision=decision)
            case ReconcileCommand():
                await ops.activate_generation(
                    self._db,
                    job_id=command.job_id,
                    target_generation=command.target_generation,
                    reconciled_at=datetime.now(UTC),
                )
                return RebuildStateOutput()
            case ReleaseCommand():
                await ops.release_claim(self._db, command.job_id)
                return RebuildStateOutput(released=True)
            case StartCommand():
                await ops.start(self._db, command.job_id)
                return RebuildStateOutput()
            case FailRebuildCommand():
                found = await ops.fail_rebuild(self._db, command.job_id, command.error)
                return RebuildStateOutput(released=found)
            case CompleteRebuildCommand():
                await ops.complete_rebuild(
                    self._db,
                    job_id=command.job_id,
                    version=command.version,
                    num_vectors=command.num_vectors,
                    dimension=command.dimension,
                    removed_versions=command.removed_versions,
                    text_num_vectors=command.text_num_vectors,
                )
                return RebuildStateOutput()
