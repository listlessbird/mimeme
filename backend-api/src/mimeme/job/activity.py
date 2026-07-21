from __future__ import annotations

from datetime import UTC, datetime, timedelta

from temporalio import activity

from mimeme.db import Db
from mimeme.job import ops
from mimeme.job.model import (
    CompleteIngestCommand,
    CompleteItemCommand,
    CompleteRebuildCommand,
    FailItemCommand,
    FailRebuildCommand,
    IngestStateCommand,
    IngestStateOutput,
    InitializeCommand,
    PrepareCommand,
    ProgressCommand,
    RebuildStateCommand,
    RebuildStateOutput,
    ReconcileCommand,
    ReleaseCommand,
    SaveInferenceCommand,
    StageCommand,
    StartCommand,
)

INGEST_STATE = "mimeme.job.ingest-state.tmp"
REBUILD_STATE = "mimeme.job.rebuild-state.tmp"


class JobActivities:
    def __init__(self, db: Db, *, rebuild_claim_timeout: timedelta) -> None:
        self._db = db
        self._claim_timeout = rebuild_claim_timeout

    @activity.defn(name=INGEST_STATE)
    async def ingest_state(self, command: IngestStateCommand) -> IngestStateOutput:
        match command:
            case InitializeCommand():
                init = await ops.initialize_ingest(self._db, command.job_id)
                return IngestStateOutput(init=init)
            case StageCommand():
                found = await ops.record_stage(self._db, command.ingest_url_id, command.stage)
                return IngestStateOutput(found=found)
            case FailItemCommand():
                found = await ops.mark_item_failed(self._db, command.ingest_url_id, command.error)
                return IngestStateOutput(found=found)
            case CompleteItemCommand():
                done = await ops.mark_item_done(
                    self._db,
                    command.ingest_url_id,
                    command.image_id,
                    duplicate_reason=command.duplicate_reason,
                    duplicate_of_image_id=command.duplicate_of_image_id,
                )
                return IngestStateOutput(found=done.found, image_exists=done.image_exists)
            case SaveInferenceCommand():
                index_changed: bool | None = None
                desired_generation: int | None = None
                if command.annotation is not None:
                    await ops.save_annotations(
                        self._db,
                        image_id=command.annotation.image_id,
                        caption=command.annotation.caption,
                        caption_model=command.annotation.caption_model,
                        ocr_text=command.annotation.ocr_text,
                        ocr_model=command.annotation.ocr_model,
                    )
                if command.embedding is not None:
                    saved = await ops.save_embedding(
                        self._db,
                        image_id=command.embedding.image_id,
                        model=command.embedding.model,
                        dimension=command.embedding.dimension,
                        image_embedding_key=command.embedding.image_embedding_key,
                    )
                    index_changed = saved.index_changed
                    desired_generation = saved.desired_generation
                return IngestStateOutput(
                    index_changed=index_changed, desired_generation=desired_generation
                )
            case ProgressCommand():
                found = await ops.progress(
                    self._db, command.job_id, command.progress, command.message
                )
                return IngestStateOutput(found=found)
            case CompleteIngestCommand():
                found = await ops.complete_ingest(
                    self._db,
                    job_id=command.job_id,
                    processed=command.processed,
                    failed=command.failed,
                    duplicates=command.duplicates,
                )
                return IngestStateOutput(found=found)

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
