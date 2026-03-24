"""Tests for workflow state management activities.

These tests use the real test database with transaction rollback isolation.
They verify that activities correctly mutate job and ingest URL state.
"""

from __future__ import annotations

import json

import pytest
from temporalio.testing import ActivityEnvironment

from activities.workflow_state.activities import (
    complete_ingest_job_activity,
    ingest_initialize_activity,
    mark_ingest_url_done_activity,
    mark_ingest_url_failed_activity,
    save_annotations_activity,
    save_embedding_info_activity,
    start_rebuild_job_activity,
    update_job_progress_activity,
)
from activities.workflow_state.models import (
    CompleteIngestJobInput,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)
from shared.models.orm import (
    Annotation,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    Processing,
    ProcessingStatus,
)
from tests.factories import ImageFactory, IngestURLFactory, JobFactory, ProcessingFactory


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ---------------------------------------------------------------------------
# ingest_initialize_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestIngestInitializeActivity:
    async def test_marks_job_running_and_returns_urls(
        self, db_session, activity_env
    ) -> None:
        job = JobFactory(session=db_session, type=JobType.INGEST)
        url1 = IngestURLFactory(session=db_session, job=job)
        url2 = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        result = await activity_env.run(ingest_initialize_activity, job.id)

        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert len(result.urls) == 2
        url_ids = {u.id for u in result.urls}
        assert url1.id in url_ids
        assert url2.id in url_ids

    async def test_nonexistent_job_raises(self, db_session, activity_env) -> None:
        with pytest.raises(ValueError, match="not found"):
            await activity_env.run(ingest_initialize_activity, "nonexistent-job")


# ---------------------------------------------------------------------------
# mark_ingest_url_done_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkIngestUrlDoneActivity:
    async def test_marks_url_done_with_image_id(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)
        await activity_env.run(mark_ingest_url_done_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id

    async def test_nonexistent_image_marks_failed(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=999999)
        await activity_env.run(mark_ingest_url_done_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED
        assert "not found" in url.error_message

    async def test_nonexistent_url_is_noop(self, db_session, activity_env) -> None:
        """Calling with a nonexistent URL ID should not raise."""
        inp = MarkIngestUrlDoneInput(ingest_url_id=999999, image_id=1)
        await activity_env.run(mark_ingest_url_done_activity, inp)


# ---------------------------------------------------------------------------
# mark_ingest_url_failed_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkIngestUrlFailedActivity:
    async def test_marks_url_failed_with_error(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="HTTP 404")
        await activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED
        assert url.error_message == "HTTP 404"

    async def test_truncates_long_error(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        long_error = "x" * 2000
        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error=long_error)
        await activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert len(url.error_message) == 1000


# ---------------------------------------------------------------------------
# save_annotations_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveAnnotationsActivity:
    async def test_creates_annotation(self, db_session, activity_env) -> None:
        image = ImageFactory(session=db_session)
        ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="A funny cat meme",
            caption_model="moondream2",
            ocr_text="IMPACT FONT TEXT",
            ocr_model="moondream2",
        )
        await activity_env.run(save_annotations_activity, inp)

        ann = db_session.query(Annotation).filter_by(image_id=image.id).first()
        assert ann is not None
        assert ann.caption_text == "A funny cat meme"
        assert ann.ocr_text == "IMPACT FONT TEXT"

    async def test_updates_processing_status(self, db_session, activity_env) -> None:
        image = ImageFactory(session=db_session)
        proc = ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="caption",
            caption_model="moondream2",
            ocr_text="text",
            ocr_model="moondream2",
        )
        await activity_env.run(save_annotations_activity, inp)

        db_session.refresh(proc)
        assert proc.caption_status == ProcessingStatus.DONE
        assert proc.ocr_status == ProcessingStatus.DONE
        assert proc.caption_model == "moondream2"

    async def test_idempotent_upsert(self, db_session, activity_env) -> None:
        """Calling save_annotations twice with same image_id should upsert, not crash."""
        image = ImageFactory(session=db_session)
        ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="first caption",
            caption_model="model-v1",
            ocr_text="first ocr",
            ocr_model="model-v1",
        )
        await activity_env.run(save_annotations_activity, inp)

        inp2 = SaveAnnotationsInput(
            image_id=image.id,
            caption="updated caption",
            caption_model="model-v2",
            ocr_text="updated ocr",
            ocr_model="model-v2",
        )
        await activity_env.run(save_annotations_activity, inp2)

        ann = db_session.query(Annotation).filter_by(image_id=image.id).first()
        assert ann.caption_text == "updated caption"
        assert ann.ocr_text == "updated ocr"


# ---------------------------------------------------------------------------
# save_embedding_info_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveEmbeddingInfoActivity:
    async def test_updates_processing_embed_fields(self, db_session, activity_env) -> None:
        image = ImageFactory(session=db_session)
        proc = ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        await activity_env.run(save_embedding_info_activity, inp)

        db_session.refresh(proc)
        assert proc.embed_status == ProcessingStatus.DONE
        assert proc.embed_model == "siglip2-base"
        assert proc.embed_dim == 768
        assert proc.embed_s3_key == "embeddings/test/abc.npy"

    async def test_idempotent_on_retry(self, db_session, activity_env) -> None:
        """Calling twice should just overwrite, not crash."""
        image = ImageFactory(session=db_session)
        proc = ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        await activity_env.run(save_embedding_info_activity, inp)
        await activity_env.run(save_embedding_info_activity, inp)

        db_session.refresh(proc)
        assert proc.embed_status == ProcessingStatus.DONE

    async def test_missing_processing_is_noop(self, db_session, activity_env) -> None:
        """If no Processing row exists, activity completes without error."""
        image = ImageFactory(session=db_session)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        await activity_env.run(save_embedding_info_activity, inp)


# ---------------------------------------------------------------------------
# update_job_progress_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestUpdateJobProgressActivity:
    async def test_updates_progress(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()

        inp = UpdateJobProgressInput(job_id=job.id, progress=50.0, message="Halfway")
        await activity_env.run(update_job_progress_activity, inp)

        db_session.refresh(job)
        assert job.progress == 50.0
        assert job.message == "Halfway"

    async def test_progress_without_message(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session, message="Old message")
        db_session.flush()

        inp = UpdateJobProgressInput(job_id=job.id, progress=75.0)
        await activity_env.run(update_job_progress_activity, inp)

        db_session.refresh(job)
        assert job.progress == 75.0
        assert job.message == "Old message"  # unchanged

    async def test_nonexistent_job_is_noop(self, db_session, activity_env) -> None:
        inp = UpdateJobProgressInput(job_id="nonexistent", progress=50.0)
        await activity_env.run(update_job_progress_activity, inp)


# ---------------------------------------------------------------------------
# complete_ingest_job_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestCompleteIngestJobActivity:
    async def test_no_failures_marks_completed(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(
            job_id=job.id, processed=5, failed=0, duplicates=1
        )
        await activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.completed_at is not None
        result = json.loads(job.result)
        assert result["processed"] == 5
        assert result["duplicates"] == 1

    async def test_with_failures_marks_failed(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(
            job_id=job.id, processed=3, failed=2, duplicates=0
        )
        await activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED

    async def test_idempotent_on_retry(self, db_session, activity_env) -> None:
        """Calling complete twice should overwrite result, not crash."""
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(
            job_id=job.id, processed=5, failed=0, duplicates=0
        )
        await activity_env.run(complete_ingest_job_activity, inp)
        await activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# start_rebuild_job_activity
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_session_scope")
class TestStartRebuildJobActivity:
    async def test_marks_job_running(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        inp = StartRebuildJobInput(job_id=job.id)
        await activity_env.run(start_rebuild_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    async def test_nonexistent_job_raises(self, db_session, activity_env) -> None:
        inp = StartRebuildJobInput(job_id="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            await activity_env.run(start_rebuild_job_activity, inp)
