from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.embedding import EmbedBatchInput, EmbedBatchOutput, EmbedImageInput
    from activities.storage import (
        DownloadImageInput,
        DownloadImageOutput,
        ProcessImageInput,
        ProcessImageOutput,
    )
    from activities.vision import CaptionInput, CaptionOutput, OCRInput, OCROutput
    from activities.workflow_state import (
        CompleteIngestJobInput,
        IngestInitOutput,
        MarkIngestUrlDoneInput,
        MarkIngestUrlFailedInput,
        SaveAnnotationsInput,
        SaveEmbeddingInfoInput,
        UpdateJobProgressInput,
    )
    from shared.config import settings
    from workflows.models import IngestWorkflowInput, IngestWorkflowOutput


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
            init = IngestInitOutput.model_validate(
                await workflow.execute_activity(
                    "ingest_initialize_activity",
                    input.job_id,
                    task_queue=settings.temporal_task_queue_cpu,
                    start_to_close_timeout=timedelta(minutes=1),
                )
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
                    download_result = DownloadImageOutput.model_validate(
                        await workflow.execute_activity(
                            "download_image_activity",
                            DownloadImageInput(
                                url=url,
                                job_id=input.job_id,
                                ingest_url_id=ingest_url_id,
                            ),
                            task_queue=settings.temporal_task_queue_cpu,
                            start_to_close_timeout=timedelta(minutes=5),
                            retry_policy=RetryPolicy(
                                maximum_attempts=3,
                                initial_interval=timedelta(seconds=1),
                                maximum_interval=timedelta(seconds=30),
                            ),
                        )
                    )

                    if not download_result.success:
                        failed += 1
                        last_step = "mark_failed_download"
                        await workflow.execute_activity(
                            "mark_ingest_url_failed_activity",
                            MarkIngestUrlFailedInput(
                                ingest_url_id=ingest_url_id,
                                error=download_result.error or "Download failed",
                            ),
                            task_queue=settings.temporal_task_queue_cpu,
                            start_to_close_timeout=timedelta(minutes=1),
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
                    process_result = ProcessImageOutput.model_validate(
                        await workflow.execute_activity(
                            "process_image_activity",
                            ProcessImageInput(
                                local_path=download_result.local_path,
                                filename=download_result.filename,
                                ingest_url_id=ingest_url_id,
                                dataset=input.dataset,
                            ),
                            task_queue=settings.temporal_task_queue_cpu,
                            start_to_close_timeout=timedelta(minutes=5),
                        )
                    )

                    if process_result.is_duplicate:
                        duplicates += 1
                        last_step = "mark_duplicate_done"
                        await workflow.execute_activity(
                            "mark_ingest_url_done_activity",
                            MarkIngestUrlDoneInput(
                                ingest_url_id=ingest_url_id,
                                image_id=process_result.image_id,
                            ),
                            task_queue=settings.temporal_task_queue_cpu,
                            start_to_close_timeout=timedelta(minutes=1),
                        )
                        continue

                    last_step = "caption"
                    workflow.logger.info(
                        "workflow_step",
                        extra={
                            "workflow_name": "IngestWorkflow",
                            "job_id": input.job_id,
                            "image_id": process_result.image_id,
                            "step": "caption_image",
                        },
                    )
                    caption_result = CaptionOutput.model_validate(
                        await workflow.execute_activity(
                            "caption_activity",
                            CaptionInput(
                                image_id=process_result.image_id,
                                s3_key=process_result.s3_key,
                            ),
                            task_queue=settings.temporal_task_queue_gpu,
                            start_to_close_timeout=timedelta(minutes=10),
                        )
                    )

                    last_step = "ocr"
                    workflow.logger.info(
                        "workflow_step",
                        extra={
                            "workflow_name": "IngestWorkflow",
                            "job_id": input.job_id,
                            "image_id": process_result.image_id,
                            "step": "ocr_image",
                        },
                    )
                    ocr_result = OCROutput.model_validate(
                        await workflow.execute_activity(
                            "ocr_activity",
                            OCRInput(
                                image_id=process_result.image_id,
                                s3_key=process_result.s3_key,
                            ),
                            task_queue=settings.temporal_task_queue_gpu,
                            start_to_close_timeout=timedelta(minutes=10),
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
                        "save_annotations_activity",
                        SaveAnnotationsInput(
                            image_id=process_result.image_id,
                            caption=caption_result.caption,
                            caption_model=caption_result.model,
                            ocr_text=ocr_result.text,
                            ocr_model=ocr_result.model,
                        ),
                        task_queue=settings.temporal_task_queue_cpu,
                        start_to_close_timeout=timedelta(minutes=1),
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
                    embed_result = EmbedBatchOutput.model_validate(
                        await workflow.execute_activity(
                            "embed_batch_activity",
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
                            task_queue=settings.temporal_task_queue_gpu,
                            start_to_close_timeout=timedelta(minutes=10),
                        )
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
                            "save_embedding_info_activity",
                            SaveEmbeddingInfoInput(
                                image_id=result.image_id,
                                model=result.model,
                                dimension=result.dimension,
                                image_embedding_key=result.image_embedding_key,
                            ),
                            task_queue=settings.temporal_task_queue_cpu,
                            start_to_close_timeout=timedelta(minutes=1),
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
                        "mark_ingest_url_done_activity",
                        MarkIngestUrlDoneInput(
                            ingest_url_id=ingest_url_id,
                            image_id=process_result.image_id,
                        ),
                        task_queue=settings.temporal_task_queue_cpu,
                        start_to_close_timeout=timedelta(minutes=1),
                    )
                    processed += 1

                except Exception as e:
                    failed += 1
                    last_step = "mark_failed_exception"
                    compact_error = _compact_error(e)
                    await workflow.execute_activity(
                        "mark_ingest_url_failed_activity",
                        MarkIngestUrlFailedInput(
                            ingest_url_id=ingest_url_id,
                            error=compact_error,
                        ),
                        task_queue=settings.temporal_task_queue_cpu,
                        start_to_close_timeout=timedelta(minutes=1),
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
                progress = (processed / total) * 100 if total > 0 else 0
                await workflow.execute_activity(
                    "update_job_progress_activity",
                    UpdateJobProgressInput(job_id=input.job_id, progress=progress),
                    task_queue=settings.temporal_task_queue_cpu,
                    start_to_close_timeout=timedelta(minutes=1),
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
                "complete_ingest_job_activity",
                CompleteIngestJobInput(
                    job_id=input.job_id,
                    processed=processed,
                    failed=failed,
                    duplicates=duplicates,
                ),
                task_queue=settings.temporal_task_queue_cpu,
                start_to_close_timeout=timedelta(minutes=1),
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
