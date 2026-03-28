from __future__ import annotations

import asyncio
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
    from activities.vision.activities import caption_activity, ocr_activity
    from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput
    from activities.workflow_state.activities import (
        complete_ingest_job_activity,
        ingest_initialize_activity,
        mark_ingest_url_done_activity,
        mark_ingest_url_failed_activity,
        save_annotations_activity,
        save_embedding_info_activity,
        update_job_progress_activity,
    )
    from activities.workflow_state.models import (
        CompleteIngestJobInput,
        IngestInitOutput,
        MarkIngestUrlDoneInput,
        MarkIngestUrlFailedInput,
        SaveAnnotationsInput,
        SaveEmbeddingInfoInput,
        UpdateJobProgressInput,
    )
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


def _compact_error(exc: Exception, limit: int = 500) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


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
            total = len(url_data)

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
                        failed += 1
                        last_step = "mark_failed_download"
                        await workflow.execute_activity(
                            mark_ingest_url_failed_activity,
                            MarkIngestUrlFailedInput(
                                ingest_url_id=ingest_url_id,
                                error=download_result.error or "Download failed",
                            ),
                            start_to_close_timeout=timedelta(minutes=1),
                            retry_policy=RETRY_DB,
                        )
                        continue

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
                        duplicates += 1
                        last_step = "mark_duplicate_done"
                        await workflow.execute_activity(
                            mark_ingest_url_done_activity,
                            MarkIngestUrlDoneInput(
                                ingest_url_id=ingest_url_id,
                                image_id=process_result.image_id,
                            ),
                            start_to_close_timeout=timedelta(minutes=1),
                            retry_policy=RETRY_DB,
                        )
                        continue

                    last_step = "caption_and_ocr"
                    workflow.logger.info(
                        "workflow_step",
                        extra={
                            "workflow_name": "IngestWorkflow",
                            "job_id": input.job_id,
                            "image_id": process_result.image_id,
                            "step": "caption_and_ocr",
                        },
                    )
                    caption_result: CaptionOutput
                    ocr_result: OCROutput
                    caption_result, ocr_result = await asyncio.gather(
                        workflow.execute_activity(
                            caption_activity,
                            CaptionInput(
                                image_id=process_result.image_id,
                                s3_key=process_result.s3_key,
                            ),
                            start_to_close_timeout=timedelta(minutes=10),
                            heartbeat_timeout=timedelta(minutes=2),
                            retry_policy=RETRY_GPU,
                        ),
                        workflow.execute_activity(
                            ocr_activity,
                            OCRInput(
                                image_id=process_result.image_id,
                                s3_key=process_result.s3_key,
                            ),
                            start_to_close_timeout=timedelta(minutes=10),
                            heartbeat_timeout=timedelta(minutes=2),
                            retry_policy=RETRY_GPU,
                        ),
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
                            caption=caption_result.caption,
                            caption_model=caption_result.model,
                            ocr_text=ocr_result.text,
                            ocr_model=ocr_result.model,
                        ),
                        start_to_close_timeout=timedelta(minutes=1),
                        retry_policy=RETRY_DB,
                    )

                    text = " ".join(filter(None, [caption_result.caption, ocr_result.text]))

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
                        heartbeat_timeout=timedelta(minutes=2),
                        retry_policy=RETRY_GPU,
                    )

                    if embed_result.results:
                        result = embed_result.results[0]
                        last_step = "save_embedding_info"
                        workflow.logger.info(
                            "workflow_step",
                            extra={
                                "workflow_name": "IngestWorkflow",
                                "job_id": input.job_id,
                                "image_id": result.image_id,
                                "step": "save_embedding_info",
                            },
                        )
                        await workflow.execute_activity(
                            save_embedding_info_activity,
                            SaveEmbeddingInfoInput(
                                image_id=result.image_id,
                                model=result.model,
                                dimension=result.dimension,
                                image_embedding_key=result.image_embedding_key,
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
                    processed += 1

                except Exception as e:
                    failed += 1
                    last_step = "mark_failed_exception"
                    compact_error = _compact_error(e)
                    await workflow.execute_activity(
                        mark_ingest_url_failed_activity,
                        MarkIngestUrlFailedInput(
                            ingest_url_id=ingest_url_id,
                            error=compact_error,
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
                progress = ((processed + failed + duplicates) / total) * 100 if total > 0 else 0
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
            await workflow.execute_activity(
                complete_ingest_job_activity,
                CompleteIngestJobInput(
                    job_id=input.job_id,
                    processed=processed,
                    failed=failed,
                    duplicates=duplicates,
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            result = IngestWorkflowOutput(
                job_id=input.job_id,
                total=total,
                processed=processed,
                failed=failed,
                duplicates=duplicates,
            )
            return result
        except Exception as exc:
            error_message = _compact_error(exc, limit=1000)
            raise
        finally:
            workflow.logger.info(
                "workflow_wide_event",
                extra={
                    "event_type": "workflow_wide_event",
                    "workflow_name": "IngestWorkflow",
                    "workflow_id": workflow.info().workflow_id,
                    "run_id": workflow.info().run_id,
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
