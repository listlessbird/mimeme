"""Tests for workflow state management activities.

These activities are the DB mutation layer used by both IngestWorkflow and
RebuildIndexWorkflow.  Tests verify correct state transitions, edge cases,
constraints, and idempotency (safe to retry after partial completion).

Uses the real test database with transaction rollback isolation.
"""

from __future__ import annotations

import datetime
import json

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.activities.workflow_state.activities import (
    complete_ingest_job_activity,
    complete_rebuild_job_activity,
    fail_rebuild_job_activity,
    ingest_initialize_activity,
    mark_ingest_url_done_activity,
    mark_ingest_url_failed_activity,
    prepare_rebuild_activity,
    reconcile_generation_activity,
    release_rebuild_claim_activity,
    save_annotations_activity,
    save_embedding_info_activity,
    start_rebuild_job_activity,
    update_job_progress_activity,
)
from mimeme.activities.workflow_state.models import (
    CompleteIngestJobInput,
    CompleteRebuildJobInput,
    FailRebuildJobInput,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    PrepareRebuildInput,
    ReconcileGenerationInput,
    ReleaseRebuildClaimInput,
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)
from mimeme.db.schema import (
    Annotation,
    Image,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    Processing,
    ProcessingStatus,
    RebuildTrigger,
    SearchIndexState,
)
from mimeme.domain.index_freshness import (
    IndexFreshness,
    RebuildClaimOwnershipError,
    SearchIndexStateMissingError,
)
from tests.factories import (
    create_image,
    create_ingest_url,
    create_job,
    create_processing,
    create_search_index_state,
)


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ==========================================================================
# ingest_initialize_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestIngestInitializeActivity:
    async def test_marks_job_running_and_returns_urls(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, type=JobType.INGEST)
        url1: IngestURL = create_ingest_url(session=db_session, job=job)
        url2: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        result = activity_env.run(ingest_initialize_activity, job.id)

        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert len(result.urls) == 2
        url_ids = {u.id for u in result.urls}
        assert url1.id in url_ids
        assert url2.id in url_ids

    async def test_nonexistent_job_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            activity_env.run(ingest_initialize_activity, "nonexistent-job")

    async def test_job_with_no_urls_returns_empty(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, type=JobType.INGEST)
        db_session.flush()

        result = activity_env.run(ingest_initialize_activity, job.id)
        assert len(result.urls) == 0
        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING


# ==========================================================================
# mark_ingest_url_done_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkIngestUrlDoneActivity:
    async def test_marks_url_done_with_image_id(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        image: Image = create_image(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)
        activity_env.run(mark_ingest_url_done_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id

    async def test_nonexistent_image_marks_failed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=999999)
        activity_env.run(mark_ingest_url_done_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED
        assert url.error_message is not None
        assert "not found" in url.error_message

    async def test_nonexistent_url_is_noop(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """Calling with a nonexistent URL ID should not raise."""
        inp = MarkIngestUrlDoneInput(ingest_url_id=999999, image_id=1)
        activity_env.run(mark_ingest_url_done_activity, inp)

    async def test_idempotent_calling_twice_is_safe(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """mark_ingest_url_done should be safe to call twice for the same URL."""
        job: Job = create_job(session=db_session)
        image: Image = create_image(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)

        activity_env.run(mark_ingest_url_done_activity, inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE

        # Call again — should not crash
        activity_env.run(mark_ingest_url_done_activity, inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id


# ==========================================================================
# mark_ingest_url_failed_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkIngestUrlFailedActivity:
    async def test_marks_url_failed_with_error(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="HTTP 404")
        activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED
        assert url.error_message == "HTTP 404"

    async def test_truncates_long_error(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        long_error = "x" * 2000
        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error=long_error)
        activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert url.error_message is not None
        assert len(url.error_message) == 1000

    async def test_idempotent_calling_twice_is_safe(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="First error")

        activity_env.run(mark_ingest_url_failed_activity, inp)
        activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED

    async def test_mark_failed_after_done_overwrites(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """If a URL was already DONE and then marked failed on retry,
        it should update to FAILED without error."""
        job: Job = create_job(session=db_session)
        image: Image = create_image(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()

        done_inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)
        activity_env.run(mark_ingest_url_done_activity, done_inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE

        fail_inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="Late failure")
        activity_env.run(mark_ingest_url_failed_activity, fail_inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED


# ==========================================================================
# save_annotations_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveAnnotationsActivity:
    async def test_creates_annotation_and_updates_processing(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        image: Image = create_image(session=db_session)
        proc: Processing = create_processing(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="A funny cat meme",
            caption_model="moondream2",
            ocr_text="IMPACT FONT TEXT",
            ocr_model="moondream2",
        )
        activity_env.run(save_annotations_activity, inp)

        ann = db_session.query(Annotation).filter_by(image_id=image.id).first()
        assert ann is not None
        assert ann.caption_text == "A funny cat meme"
        assert ann.ocr_text == "IMPACT FONT TEXT"

        db_session.refresh(proc)
        assert proc.caption_status == ProcessingStatus.DONE
        assert proc.ocr_status == ProcessingStatus.DONE
        assert proc.caption_model == "moondream2"

    async def test_idempotent_upsert_no_duplicate_rows(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """Calling save_annotations twice should upsert, not create duplicates."""
        image: Image = create_image(session=db_session)
        create_processing(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="first caption",
            caption_model="model-v1",
            ocr_text="first ocr",
            ocr_model="model-v1",
        )
        activity_env.run(save_annotations_activity, inp)

        inp2 = SaveAnnotationsInput(
            image_id=image.id,
            caption="updated caption",
            caption_model="model-v2",
            ocr_text="updated ocr",
            ocr_model="model-v2",
        )
        activity_env.run(save_annotations_activity, inp2)

        count = db_session.query(Annotation).filter_by(image_id=image.id).count()
        assert count == 1

        ann = db_session.query(Annotation).filter_by(image_id=image.id).first()
        assert ann is not None
        assert ann.caption_text == "updated caption"
        assert ann.ocr_text == "updated ocr"


# ==========================================================================
# save_embedding_info_activity
# ==========================================================================


def _desired_generation(session: Session) -> int:
    return IndexFreshness(session).get().desired_generation


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveEmbeddingInfoActivity:
    async def test_updates_processing_embed_fields(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session)
        image: Image = create_image(session=db_session)
        proc: Processing = create_processing(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        activity_env.run(save_embedding_info_activity, inp)

        db_session.refresh(proc)
        assert proc.embed_status == ProcessingStatus.DONE
        assert proc.embed_model == "siglip2-base"
        assert proc.embed_dim == 768
        assert proc.embed_s3_key == "embeddings/test/abc.npy"

    async def test_new_embedding_increments_desired_generation_once(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session, desired_generation=1, active_generation=1)
        image: Image = create_image(session=db_session)
        create_processing(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        activity_env.run(save_embedding_info_activity, inp)

        assert _desired_generation(db_session) == 2

    async def test_idempotent_on_retry(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session, desired_generation=1, active_generation=1)
        image: Image = create_image(session=db_session)
        proc: Processing = create_processing(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        activity_env.run(save_embedding_info_activity, inp)
        activity_env.run(save_embedding_info_activity, inp)

        db_session.refresh(proc)
        assert proc.embed_status == ProcessingStatus.DONE
        assert _desired_generation(db_session) == 2

    async def test_embedding_repair_increments_generation(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session, desired_generation=5, active_generation=5)
        image: Image = create_image(session=db_session)
        create_processing(
            session=db_session,
            image=image,
            embed_status=ProcessingStatus.DONE,
            embed_model="siglip2-base",
            embed_dim=768,
            embed_s3_key="embeddings/test/abc.npy",
        )
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=1024,
            image_embedding_key="embeddings/test/abc.npy",
        )
        activity_env.run(save_embedding_info_activity, inp)

        assert _desired_generation(db_session) == 6

    async def test_missing_processing_does_not_increment(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session, desired_generation=3, active_generation=3)
        image: Image = create_image(session=db_session)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/test/abc.npy",
        )
        activity_env.run(save_embedding_info_activity, inp)

        assert _desired_generation(db_session) == 3


# ==========================================================================
# update_job_progress_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestUpdateJobProgressActivity:
    async def test_updates_progress_and_message(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session)
        db_session.flush()

        inp = UpdateJobProgressInput(job_id=job.id, progress=50.0, message="Halfway")
        activity_env.run(update_job_progress_activity, inp)

        db_session.refresh(job)
        assert job.progress == 50.0
        assert job.message == "Halfway"

    async def test_progress_without_message_preserves_old(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, message="Old message")
        db_session.flush()

        inp = UpdateJobProgressInput(job_id=job.id, progress=75.0)
        activity_env.run(update_job_progress_activity, inp)

        db_session.refresh(job)
        assert job.progress == 75.0
        assert job.message == "Old message"

    async def test_nonexistent_job_is_noop(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        inp = UpdateJobProgressInput(job_id="nonexistent", progress=50.0)
        activity_env.run(update_job_progress_activity, inp)


# ==========================================================================
# complete_ingest_job_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestCompleteIngestJobActivity:
    async def test_no_failures_marks_completed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(job_id=job.id, processed=5, failed=0, duplicates=1)
        activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.completed_at is not None
        assert job.result is not None
        result = json.loads(job.result)
        assert result["processed"] == 5
        assert result["duplicates"] == 1

    async def test_with_failures_marks_failed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(job_id=job.id, processed=3, failed=2, duplicates=0)
        activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED

    async def test_idempotent_on_retry(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """Completing a job twice should overwrite result, not crash."""
        job: Job = create_job(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(job_id=job.id, processed=5, failed=0, duplicates=1)
        activity_env.run(complete_ingest_job_activity, inp)
        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED

        activity_env.run(complete_ingest_job_activity, inp)
        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED


# ==========================================================================
# start_rebuild_job_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestStartRebuildJobActivity:
    async def test_marks_job_running(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        inp = StartRebuildJobInput(job_id=job.id)
        activity_env.run(start_rebuild_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    async def test_nonexistent_job_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        inp = StartRebuildJobInput(job_id="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            activity_env.run(start_rebuild_job_activity, inp)


# ==========================================================================
# fail_rebuild_job_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestFailRebuildJobActivity:
    async def test_marks_job_failed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(
            session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING
        )
        db_session.flush()

        inp = FailRebuildJobInput(job_id=job.id, error="No embeddings found")
        activity_env.run(fail_rebuild_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED
        assert job.message == "No embeddings found"
        assert job.completed_at is not None

    async def test_truncates_long_error(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(
            session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING
        )
        db_session.flush()

        inp = FailRebuildJobInput(job_id=job.id, error="x" * 5000)
        activity_env.run(fail_rebuild_job_activity, inp)

        db_session.refresh(job)
        assert job.message is not None
        assert len(job.message) == 2000


# ==========================================================================
# complete_rebuild_job_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestCompleteRebuildJobActivity:
    async def test_marks_job_completed_with_result(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(
            session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING
        )
        db_session.flush()

        inp = CompleteRebuildJobInput(
            job_id=job.id,
            version="v-abc123",
            num_vectors=500,
            dimension=768,
            removed_versions=["v-old-001"],
            text_num_vectors=500,
        )
        activity_env.run(complete_rebuild_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.completed_at is not None
        assert job.result is not None
        result = json.loads(job.result)
        assert result["version"] == "v-abc123"
        assert result["num_vectors"] == 500
        assert result["removed_versions"] == ["v-old-001"]

    async def test_nonexistent_job_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        inp = CompleteRebuildJobInput(
            job_id="nonexistent",
            version="v-1",
            num_vectors=100,
            dimension=768,
            removed_versions=[],
        )
        with pytest.raises(ValueError, match="not found"):
            activity_env.run(complete_rebuild_job_activity, inp)


# ==========================================================================
# prepare_rebuild_activity / reconcile_generation_activity /
# release_rebuild_claim_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestPrepareRebuildActivity:
    async def test_manual_dirty_claims_and_returns_build(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_search_index_state(session=db_session, desired_generation=4, active_generation=1)
        job: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        inp = PrepareRebuildInput(
            job_id=job.id, workflow_id="wf-1", force=False, trigger=RebuildTrigger.MANUAL
        )
        result = activity_env.run(prepare_rebuild_activity, inp)

        assert result.decision == "build"
        assert result.job_id == job.id
        assert result.target_generation == 4
        state = db_session.get(SearchIndexState, 1)
        assert state is not None and state.rebuild_job_id == job.id

    async def test_missing_state_row_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        inp = PrepareRebuildInput(
            job_id=None, workflow_id="wf-1", force=False, trigger=RebuildTrigger.SCHEDULED
        )
        with pytest.raises(SearchIndexStateMissingError):
            activity_env.run(prepare_rebuild_activity, inp)


@pytest.mark.usefixtures("_patch_session_scope")
class TestReconcileGenerationActivity:
    async def test_advances_active_generation(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        create_search_index_state(
            session=db_session,
            desired_generation=5,
            active_generation=1,
            rebuild_job_id=job.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
        )

        inp = ReconcileGenerationInput(job_id=job.id, target_generation=5)
        activity_env.run(reconcile_generation_activity, inp)

        state = db_session.get(SearchIndexState, 1)
        assert state is not None and state.active_generation == 5
        assert state.last_reconciled_at is not None

    async def test_wrong_owner_propagates(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        owner: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        other: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        create_search_index_state(
            session=db_session,
            desired_generation=5,
            active_generation=1,
            rebuild_job_id=owner.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
        )

        inp = ReconcileGenerationInput(job_id=other.id, target_generation=5)
        with pytest.raises(RebuildClaimOwnershipError):
            activity_env.run(reconcile_generation_activity, inp)


@pytest.mark.usefixtures("_patch_session_scope")
class TestReleaseRebuildClaimActivity:
    async def test_releases_own_claim(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        job: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        create_search_index_state(
            session=db_session,
            desired_generation=5,
            active_generation=1,
            rebuild_job_id=job.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
        )

        activity_env.run(release_rebuild_claim_activity, ReleaseRebuildClaimInput(job_id=job.id))

        state = db_session.get(SearchIndexState, 1)
        assert state is not None and state.rebuild_job_id is None

    async def test_foreign_claim_is_left_intact_without_raising(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        owner: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        other: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        create_search_index_state(
            session=db_session,
            desired_generation=5,
            active_generation=1,
            rebuild_job_id=owner.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
        )

        activity_env.run(release_rebuild_claim_activity, ReleaseRebuildClaimInput(job_id=other.id))

        state = db_session.get(SearchIndexState, 1)
        assert state is not None and state.rebuild_job_id == owner.id

    async def test_missing_state_row_does_not_raise(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        activity_env.run(
            release_rebuild_claim_activity, ReleaseRebuildClaimInput(job_id="rebuild-x")
        )


# ==========================================================================
# Job model constraints and defaults (tested via activities above, but also
# directly to catch ORM-level issues)
# ==========================================================================


class TestJobModelConstraints:
    def test_default_status_is_pending(self, db_session: Session) -> None:
        job: Job = create_job(session=db_session)
        db_session.flush()
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.started_at is None
        assert job.completed_at is None

    def test_duplicate_job_id_raises(self, db_session: Session) -> None:
        create_job(session=db_session, id="dupe-id")
        db_session.flush()

        job2 = Job(id="dupe-id", type=JobType.INGEST)
        db_session.add(job2)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_result_stores_valid_json(self, db_session: Session) -> None:
        job: Job = create_job(session=db_session)
        job.result = json.dumps({"key": "value", "nested": {"a": 1}})
        db_session.flush()

        db_session.refresh(job)
        result = job.result
        assert result is not None
        parsed = json.loads(result)
        assert parsed["nested"]["a"] == 1

    def test_both_job_types(self, db_session: Session) -> None:
        j1: Job = create_job(session=db_session, type=JobType.INGEST)
        j2: Job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        assert j1.type == JobType.INGEST
        assert j2.type == JobType.REBUILD_INDEX


# ==========================================================================
# IngestURL model relationships (tested via activities above, but also
# directly to verify cascade/FK config)
# ==========================================================================


class TestIngestURLModelRelationships:
    def test_multiple_urls_per_job(self, db_session: Session) -> None:
        job: Job = create_job(session=db_session)
        create_ingest_url(session=db_session, job=job)
        create_ingest_url(session=db_session, job=job)
        create_ingest_url(session=db_session, job=job)
        db_session.flush()

        db_session.refresh(job)
        assert len(job.ingest_urls) == 3

    def test_cascade_delete_job_removes_urls(self, db_session: Session) -> None:
        job: Job = create_job(session=db_session)
        create_ingest_url(session=db_session, job=job)
        create_ingest_url(session=db_session, job=job)
        db_session.flush()

        job_id = job.id
        db_session.delete(job)
        db_session.flush()

        remaining = db_session.query(IngestURL).filter_by(job_id=job_id).all()
        assert len(remaining) == 0

    def test_image_fk_set_null_on_delete(self, db_session: Session) -> None:
        """When an Image is deleted, IngestURL.image_id should be set to NULL."""
        job: Job = create_job(session=db_session)
        image: Image = create_image(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        url.image_id = image.id
        url.status = ProcessingStatus.DONE
        db_session.flush()

        db_session.delete(image)
        db_session.flush()

        db_session.refresh(url)
        assert url.image_id is None

    def test_default_status_is_pending(self, db_session: Session) -> None:
        job: Job = create_job(session=db_session)
        url: IngestURL = create_ingest_url(session=db_session, job=job)
        db_session.flush()
        assert url.status == ProcessingStatus.PENDING
