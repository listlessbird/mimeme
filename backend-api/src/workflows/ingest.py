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
    from shared.config import settings
    from shared.db import session_scope
    from shared.models import (
        Annotation,
        IngestURL,
        Job,
        JobStatus,
        Processing,
        ProcessingStatus,
    )
    from workflows.models import IngestWorkflowInput, IngestWorkflowOutput


@workflow.defn
class IngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestWorkflowOutput:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            if not job:
                raise ValueError(f"Job {input.job_id} not found")

            job.status = JobStatus.RUNNING
            session.commit()

            urls = session.query(IngestURL).filter_by(job_id=input.job_id).all()
            url_data = [(u.id, u.url) for u in urls]

        total = len(url_data)
        processed = 0
        failed = 0
        duplicates = 0

        for ingest_url_id, url in url_data:
            try:
                download_result: DownloadImageOutput = await workflow.execute_activity(
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

                if not download_result.success:
                    failed += 1
                    self._mark_url_failed(ingest_url_id, download_result.error or "Download failed")
                    continue

                process_result: ProcessImageOutput = await workflow.execute_activity(
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

                if process_result.is_duplicate:
                    duplicates += 1
                    self._mark_url_done(ingest_url_id, process_result.image_id)
                    continue

                caption_result: CaptionOutput = await workflow.execute_activity(
                    "caption_activity",
                    CaptionInput(
                        image_id=process_result.image_id,
                        s3_key=process_result.s3_key,
                    ),
                    task_queue=settings.temporal_task_queue_gpu,
                    start_to_close_timeout=timedelta(minutes=10),
                )

                ocr_result: OCROutput = await workflow.execute_activity(
                    "ocr_activity",
                    OCRInput(
                        image_id=process_result.image_id,
                        s3_key=process_result.s3_key,
                    ),
                    task_queue=settings.temporal_task_queue_gpu,
                    start_to_close_timeout=timedelta(minutes=10),
                )

                self._save_annotations(
                    process_result.image_id,
                    caption_result,
                    ocr_result,
                )

                text = " ".join(filter(None, [caption_result.caption, ocr_result.text]))

                embed_result: EmbedBatchOutput = await workflow.execute_activity(
                    "embed_batch_activity",
                    EmbedBatchInput(
                        items=[
                            EmbedImageInput(
                                image_id=process_result.image_id,
                                s3_key=process_result.s3_key,
                                text=text,
                            )
                        ],
                        dataset=input.dataset,
                    ),
                    task_queue=settings.temporal_task_queue_gpu,
                    start_to_close_timeout=timedelta(minutes=10),
                )

                if embed_result.results:
                    self._save_embedding_info(embed_result.results[0])

                self._mark_url_done(ingest_url_id, process_result.image_id)
                processed += 1

            except Exception as e:
                failed += 1
                self._mark_url_failed(ingest_url_id, str(e))

            self._update_job_progress(input.job_id, processed, total)

        self._complete_job(input.job_id, processed, failed, duplicates)

        return IngestWorkflowOutput(
            job_id=input.job_id,
            total=total,
            processed=processed,
            failed=failed,
            duplicates=duplicates,
        )

    def _mark_url_failed(self, ingest_url_id: int, error: str) -> None:
        with session_scope() as session:
            url = session.query(IngestURL).filter_by(id=ingest_url_id).first()
            if url:
                url.status = ProcessingStatus.FAILED
                url.error_message = error

    def _mark_url_done(self, ingest_url_id: int, image_id: int) -> None:
        with session_scope() as session:
            url = session.query(IngestURL).filter_by(id=ingest_url_id).first()
            if url:
                url.status = ProcessingStatus.DONE
                url.image_id = image_id

    def _save_annotations(self, image_id: int, caption: CaptionOutput, ocr: OCROutput) -> None:
        with session_scope() as session:
            ann = session.query(Annotation).filter_by(image_id=image_id).first()
            if not ann:
                ann = Annotation(image_id=image_id)
                session.add(ann)

            ann.caption_text = caption.caption
            ann.ocr_text = ocr.text

            proc = session.query(Processing).filter_by(image_id=image_id).first()
            if proc:
                proc.caption_status = ProcessingStatus.DONE
                proc.caption_model = caption.model
                proc.ocr_status = ProcessingStatus.DONE
                proc.ocr_model = ocr.model

    def _save_embedding_info(self, result) -> None:
        from activities.embedding import EmbedImageOutput

        result: EmbedImageOutput
        with session_scope() as session:
            proc = session.query(Processing).filter_by(image_id=result.image_id).first()
            if proc:
                proc.embed_status = ProcessingStatus.DONE
                proc.embed_model = result.model
                proc.embed_dim = result.dimension
                proc.embed_s3_key = result.image_embedding_key

    def _update_job_progress(self, job_id: str, processed: int, total: int) -> None:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                job.progress = (processed / total) * 100 if total > 0 else 0

    def _complete_job(self, job_id: str, processed: int, failed: int, duplicates: int) -> None:
        import json
        from datetime import UTC, datetime

        with session_scope() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                job.status = JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED
                job.progress = 100.0
                job.completed_at = datetime.now(UTC)
                job.result = json.dumps(
                    {
                        "processed": processed,
                        "failed": failed,
                        "duplicates": duplicates,
                    }
                )
