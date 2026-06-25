from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.embedding.activity import embed_batch_activity
    from activities.embedding.models import EmbedBatchInput, EmbedBatchOutput, EmbedImageInput
    from activities.storage.activities import download_image_activity, process_image_activity
    from activities.storage.models import (
        DownloadImageInput,
        DownloadImageOutput,
        ProcessImageInput,
        ProcessImageOutput,
    )
    from activities.vision.activities import annotate_image_activity
    from activities.vision.models import AnnotateImageInput, AnnotateImageOutput
    from activities.workflow_state.activities import (
        complete_ingest_job_activity,
        ingest_initialize_activity,
        mark_ingest_url_done_activity,
        mark_ingest_url_failed_activity,
        record_ingest_stage_activity,
        save_annotations_activity,
        save_embedding_info_activity,
        update_job_progress_activity,
    )
    from activities.workflow_state.models import (
        CompleteIngestJobInput,
        IngestInitOutput,
        MarkIngestUrlDoneInput,
        MarkIngestUrlFailedInput,
        RecordIngestStageInput,
        SaveAnnotationsInput,
        SaveEmbeddingInfoInput,
        UpdateJobProgressInput,
    )
    from domain.ingest_policy import IngestPolicy
    from shared.models import IngestStage
    from workflows.models import IngestWorkflowInput, IngestWorkflowOutput

RETRY_GPU = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=1),
)

RETRY_NETWORK = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
)

RETRY_DB = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn
class IngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestWorkflowOutput:
        total = 0
        processed = 0
        failed = 0
        duplicates = 0
        last_step = "initialize"
        error_message: str | None = None
        workflow_info = workflow.info()

        try:
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "IngestWorkflow",
                    "job_id": input.job_id,
                    "step": "initialize_job",
                },
            )
            init: IngestInitOutput = await workflow.execute_activity(
                ingest_initialize_activity,
                input.job_id,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            url_data = [(u.id, u.url) for u in init.urls]
            policy = IngestPolicy(total=len(url_data))
            total = policy.total

            async def record_stage(ingest_url_id: int, stage: IngestStage) -> None:
                await workflow.execute_activity(
                    record_ingest_stage_activity,
                    RecordIngestStageInput(ingest_url_id=ingest_url_id, stage=stage),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=RETRY_DB,
                )

            for ingest_url_id, url in url_data:
                try:
                    last_step = "download"
                    workflow.logger.info(
                        "workflow_step",
                        extra={
                            "workflow_name": "IngestWorkflow",
                            "job_id": input.job_id,
                            "ingest_url_id": ingest_url_id,
                            "step": "download_image",
                        },
                    )
                    await record_stage(ingest_url_id, IngestStage.DOWNLOADING)
                    download_result: DownloadImageOutput = await workflow.execute_activity(
                        download_image_activity,
                        DownloadImageInput(
                            url=url,
                            job_id=input.job_id,
                            ingest_url_id=ingest_url_id,
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=RETRY_NETWORK,
                    )

                    if not download_result.success:
                        failure = policy.record_download_failure(download_result.error)
                        last_step = "mark_failed_download"
                        await workflow.execute_activity(
                            mark_ingest_url_failed_activity,
                            MarkIngestUrlFailedInput(
                                ingest_url_id=ingest_url_id,
                                error=failure.error,
                            ),
                            start_to_close_timeout=timedelta(minutes=1),
                            retry_policy=RETRY_DB,
                        )
                    else:
                        last_step = "process_image"
                        workflow.logger.info(
                            "workflow_step",
                            extra={
                                "workflow_name": "IngestWorkflow",
                                "job_id": input.job_id,
                                "ingest_url_id": ingest_url_id,
                                "step": "process_image",
                            },
                        )
                        await record_stage(ingest_url_id, IngestStage.PROCESSING)
                        process_result: ProcessImageOutput = await workflow.execute_activity(
                            process_image_activity,
                            ProcessImageInput(
                                local_path=download_result.local_path,
                                filename=download_result.filename,
                                ingest_url_id=ingest_url_id,
                                dataset=input.dataset,
                            ),
                            start_to_close_timeout=timedelta(minutes=5),
                            retry_policy=RETRY_DB,
                        )

                        if process_result.is_duplicate:
                            duplicate = policy.record_duplicate(process_result.image_id)
                            last_step = "mark_duplicate_done"
                            await record_stage(ingest_url_id, IngestStage.DEDUPED)
                            await workflow.execute_activity(
                                mark_ingest_url_done_activity,
                                MarkIngestUrlDoneInput(
                                    ingest_url_id=ingest_url_id,
                                    image_id=duplicate.image_id,
                                    duplicate_reason=process_result.duplicate_reason,
                                    duplicate_of_image_id=process_result.duplicate_of_image_id,
                                ),
                                start_to_close_timeout=timedelta(minutes=1),
                                retry_policy=RETRY_DB,
                            )
                        else:
                            last_step = "annotate_image"
                            workflow.logger.info(
                                "workflow_step",
                                extra={
                                    "workflow_name": "IngestWorkflow",
                                    "job_id": input.job_id,
                                    "image_id": process_result.image_id,
                                    "step": "annotate_image",
                                },
                            )
                            await record_stage(ingest_url_id, IngestStage.ANNOTATING)
                            annotation_result: AnnotateImageOutput = (
                                await workflow.execute_activity(
                                    annotate_image_activity,
                                    AnnotateImageInput(
                                        image_id=process_result.image_id,
                                        s3_key=process_result.s3_key,
                                    ),
                                    start_to_close_timeout=timedelta(minutes=10),
                                    retry_policy=RETRY_GPU,
                                )
                            )

                            last_step = "save_annotations"
                            workflow.logger.info(
                                "workflow_step",
                                extra={
                                    "workflow_name": "IngestWorkflow",
                                    "job_id": input.job_id,
                                    "image_id": process_result.image_id,
                                    "step": "save_annotations",
                                },
                            )
                            await workflow.execute_activity(
                                save_annotations_activity,
                                SaveAnnotationsInput(
                                    image_id=process_result.image_id,
                                    caption=annotation_result.caption,
                                    caption_model=annotation_result.caption_model,
                                    ocr_text=annotation_result.ocr_text,
                                    ocr_model=annotation_result.ocr_model,
                                ),
                                start_to_close_timeout=timedelta(minutes=1),
                                retry_policy=RETRY_DB,
                            )

                            text = policy.compose_embedding_text(
                                annotation_result.caption,
                                annotation_result.ocr_text,
                            )

                            last_step = "embed"
                            workflow.logger.info(
                                "workflow_step",
                                extra={
                                    "workflow_name": "IngestWorkflow",
                                    "job_id": input.job_id,
                                    "image_id": process_result.image_id,
                                    "step": "embed_image",
                                },
                            )
                            await record_stage(ingest_url_id, IngestStage.EMBEDDING)
                            embed_result: EmbedBatchOutput = await workflow.execute_activity(
                                embed_batch_activity,
                                EmbedBatchInput(
                                    items=[
                                        EmbedImageInput(
                                            image_id=process_result.image_id,
                                            s3_key=process_result.s3_key,
                                            text=text,
                                            sha256=process_result.sha256,
                                            dataset=input.dataset,
                                        )
                                    ],
                                    dataset=input.dataset,
                                ),
                                start_to_close_timeout=timedelta(minutes=10),
                                retry_policy=RETRY_GPU,
                            )

                            embedding = policy.embedding_decision(embed_result.results)
                            if embedding:
                                last_step = "save_embedding_info"
                                workflow.logger.info(
                                    "workflow_step",
                                    extra={
                                        "workflow_name": "IngestWorkflow",
                                        "job_id": input.job_id,
                                        "image_id": embedding.image_id,
                                        "step": "save_embedding_info",
                                    },
                                )
                                await workflow.execute_activity(
                                    save_embedding_info_activity,
                                    SaveEmbeddingInfoInput(
                                        image_id=embedding.image_id,
                                        model=embedding.model,
                                        dimension=embedding.dimension,
                                        image_embedding_key=embedding.image_embedding_key,
                                    ),
                                    start_to_close_timeout=timedelta(minutes=1),
                                    retry_policy=RETRY_DB,
                                )

                            last_step = "mark_done"
                            workflow.logger.info(
                                "workflow_step",
                                extra={
                                    "workflow_name": "IngestWorkflow",
                                    "job_id": input.job_id,
                                    "ingest_url_id": ingest_url_id,
                                    "image_id": process_result.image_id,
                                    "step": "mark_url_done",
                                },
                            )
                            await workflow.execute_activity(
                                mark_ingest_url_done_activity,
                                MarkIngestUrlDoneInput(
                                    ingest_url_id=ingest_url_id,
                                    image_id=process_result.image_id,
                                ),
                                start_to_close_timeout=timedelta(minutes=1),
                                retry_policy=RETRY_DB,
                            )
                            await record_stage(ingest_url_id, IngestStage.COMPLETE)
                            policy.record_success(process_result.image_id)

                except Exception as e:
                    failure = policy.record_exception(e)
                    last_step = "mark_failed_exception"
                    await workflow.execute_activity(
                        mark_ingest_url_failed_activity,
                        MarkIngestUrlFailedInput(
                            ingest_url_id=ingest_url_id,
                            error=failure.error,
                        ),
                        start_to_close_timeout=timedelta(minutes=1),
                        retry_policy=RETRY_DB,
                    )

                last_step = "update_progress"
                workflow.logger.info(
                    "workflow_step",
                    extra={
                        "workflow_name": "IngestWorkflow",
                        "job_id": input.job_id,
                        "step": "update_progress",
                    },
                )
                progress = policy.progress_state().progress
                await workflow.execute_activity(
                    update_job_progress_activity,
                    UpdateJobProgressInput(job_id=input.job_id, progress=progress),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=RETRY_DB,
                )

            last_step = "complete_job"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "IngestWorkflow",
                    "job_id": input.job_id,
                    "step": "complete_job",
                },
            )
            completion = policy.completion(input.job_id)
            await workflow.execute_activity(
                complete_ingest_job_activity,
                CompleteIngestJobInput.model_construct(
                    job_id=completion.job_id,
                    processed=completion.processed,
                    failed=completion.failed,
                    duplicates=completion.duplicates,
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            result = IngestWorkflowOutput(
                job_id=completion.job_id,
                total=completion.total,
                processed=completion.processed,
                failed=completion.failed,
                duplicates=completion.duplicates,
            )
            return result
        except Exception as exc:
            message = str(exc).replace("\n", " ").strip()
            error_message = message if len(message) <= 1000 else message[:997] + "..."
            raise
        finally:
            if "policy" in locals():
                state = policy.progress_state()
                processed = state.processed
                failed = state.failed
                duplicates = state.duplicates
            try:
                workflow.logger.info(
                    "workflow_wide_event",
                    extra={
                        "event_type": "workflow_wide_event",
                        "workflow_name": "IngestWorkflow",
                        "workflow_id": workflow_info.workflow_id,
                        "run_id": workflow_info.run_id,
                        "job_id": input.job_id,
                        "dataset": input.dataset,
                        "total": total,
                        "processed": processed,
                        "failed": failed,
                        "duplicates": duplicates,
                        "last_step": last_step,
                        "outcome": "error"
                        if error_message
                        else ("failed" if failed > 0 else "success"),
                        "error": error_message,
                    },
                )
            except Exception:
                pass
