"""Tests for indexing activities: build_index, swap_index, garbage_collect.

These activities interact with FaissIndexManager, S3, and DB.  Tests mock the
heavy dependencies and verify correct orchestration, DB state, and error handling.

Note: These are sync activities (not async), so activity_env.run() returns
the result directly without needing await.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from activities.indexing.activities import (
    build_index_activity,
    garbage_collect_indexes_activity,
    swap_index_activity,
)
from activities.indexing.models import BuildIndexInput, SwapIndexInput
from shared.models import IndexBuild
from shared.models.orm import ProcessingStatus
from tests.factories import create_image, create_index_build, create_processing


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ==========================================================================
# build_index_activity
# ==========================================================================


def _done_processing(
    db_session: Session, *, model: str, count: int, s3_key: str | None = None
) -> list[str]:
    keys: list[str] = []
    for _ in range(count):
        image = create_image(session=db_session)
        proc = create_processing(session=db_session, image=image)
        proc.embed_status = ProcessingStatus.DONE
        proc.embed_model = model
        key = f"embeddings/{image.id}.npy" if s3_key is None else s3_key
        proc.embed_s3_key = key
        keys.append(key)
    db_session.flush()
    return keys


def _mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.download_numpy.return_value = np.random.rand(768).astype(np.float32)
    storage.build_index_key.return_value = "indexes/v-test/index.faiss"
    return storage


def _mock_manager(version: str = "v-test-001") -> MagicMock:
    build_result = MagicMock()
    build_result.version = version
    build_result.text_num_vectors = None
    build_result.text_s3_key = None
    manager = MagicMock()
    manager.build_index.return_value = build_result
    return manager


@contextmanager
def _patched_deps(storage: MagicMock, manager: MagicMock) -> Iterator[None]:
    with (
        patch(
            "activities.indexing.activities.get_artifact_storage_service",
            return_value=storage,
        ),
        patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=manager,
        ),
    ):
        yield


@pytest.mark.usefixtures("_patch_session_scope")
class TestBuildIndexActivity:
    def test_builds_index_from_embeddings(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        _done_processing(db_session, model="siglip2-base", count=3)
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", index_type="flat", target_generation=5)

        with _patched_deps(_mock_storage(), manager):
            result = activity_env.run(build_index_activity, inp)

        assert result.outcome == "built"
        assert result.version == "v-test-001"
        assert result.num_vectors == 3
        assert result.dimension == 768
        manager.build_index.assert_called_once()
        assert manager.build_index.call_args.kwargs["source_generation"] == 5

    def test_excludes_other_embedding_model(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        _done_processing(db_session, model="siglip2-base", count=2)
        _done_processing(db_session, model="other-model", count=3)

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(_mock_storage(), _mock_manager()):
            result = activity_env.run(build_index_activity, inp)

        assert result.num_vectors == 2

    def test_empty_corpus_no_active_index_reconciles(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(_mock_storage(), manager):
            result = activity_env.run(build_index_activity, inp)

        assert result.outcome == "empty_reconcile"
        assert result.num_vectors == 0
        manager.build_index.assert_not_called()
        manager.build_empty_index.assert_not_called()

    def test_blank_s3_key_is_not_a_candidate(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        _done_processing(db_session, model="siglip2-base", count=2, s3_key="")
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(_mock_storage(), manager):
            result = activity_env.run(build_index_activity, inp)

        assert result.outcome == "empty_reconcile"
        manager.build_index.assert_not_called()

    def test_empty_corpus_with_active_index_builds_empty(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_index_build(
            session=db_session, is_active=True, dimension=768, index_type="flat", num_vectors=10
        )
        db_session.flush()

        empty_result = MagicMock()
        empty_result.version = "v-empty-001"
        manager = _mock_manager()
        manager.build_empty_index.return_value = empty_result

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(_mock_storage(), manager):
            result = activity_env.run(build_index_activity, inp)

        assert result.outcome == "built"
        assert result.num_vectors == 0
        assert result.dimension == 768
        assert result.version == "v-empty-001"
        manager.build_empty_index.assert_called_once()
        assert manager.build_empty_index.call_args.kwargs["dimension"] == 768

    def test_empty_corpus_with_unknown_active_dimension_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_index_build(
            session=db_session, is_active=True, dimension=None, index_type="flat", num_vectors=10
        )
        db_session.flush()
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(_mock_storage(), manager):
            with pytest.raises(ValueError, match="unknown dimension"):
                activity_env.run(build_index_activity, inp)

        manager.build_empty_index.assert_not_called()

    def test_emits_numeric_stage_timings(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        _done_processing(db_session, model="siglip2-base", count=2)

        captured: dict[str, object] = {}

        def _capture(**kwargs: object) -> None:
            captured.update(kwargs)

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with (
            _patched_deps(_mock_storage(), _mock_manager()),
            patch("activities.indexing.activities.emit_activity_event", _capture),
        ):
            activity_env.run(build_index_activity, inp)

        assert captured["target_generation"] == 5
        for field in (
            "candidate_query_ms",
            "embedding_download_ms",
            "image_index_build_ms",
            "total_duration_ms",
        ):
            value = captured[field]
            assert isinstance(value, int | float)
            assert value >= 0

    def test_missing_s3_vectors_are_skipped(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        from botocore.exceptions import ClientError

        keys = _done_processing(db_session, model="siglip2-base", count=3)
        missing_key = keys[0]

        def _download_numpy(key: str) -> np.ndarray:
            if key == missing_key:
                raise ClientError({"Error": {"Code": "404"}}, "GetObject")
            return np.random.rand(768).astype(np.float32)

        storage = _mock_storage()
        storage.download_numpy.side_effect = _download_numpy

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(storage, _mock_manager("v-test-002")):
            result = activity_env.run(build_index_activity, inp)

        assert result.num_vectors == 2

    def test_all_downloads_missing_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        from botocore.exceptions import ClientError

        _done_processing(db_session, model="siglip2-base", count=3)

        storage = _mock_storage()
        storage.download_numpy.side_effect = ClientError({"Error": {"Code": "404"}}, "GetObject")
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(storage, manager):
            with pytest.raises(ValueError, match="Failed to load any embeddings"):
                activity_env.run(build_index_activity, inp)

        manager.build_index.assert_not_called()

    def test_image_dimension_mismatch_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        keys = _done_processing(db_session, model="siglip2-base", count=2)
        dims = {keys[0]: 768, keys[1]: 512}

        def _download_numpy(key: str) -> np.ndarray:
            if key.endswith("_text.npy"):
                raise RuntimeError("no text embedding")
            return np.random.rand(dims[key]).astype(np.float32)

        storage = _mock_storage()
        storage.download_numpy.side_effect = _download_numpy
        manager = _mock_manager()

        inp = BuildIndexInput(model_name="siglip2-base", target_generation=5)

        with _patched_deps(storage, manager):
            with pytest.raises(ValueError, match="dimension mismatch"):
                activity_env.run(build_index_activity, inp)

        manager.build_index.assert_not_called()


# ==========================================================================
# swap_index_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestSwapIndexActivity:
    def test_delegates_to_manager(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()

        inp = SwapIndexInput(version="v-test-001", job_id="rebuild-1", target_generation=7)

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            activity_env.run(swap_index_activity, inp)

        mock_manager.swap_to_version.assert_called_once()
        call_args = mock_manager.swap_to_version.call_args
        assert call_args[0][0] == "v-test-001"
        assert call_args.kwargs["job_id"] == "rebuild-1"
        assert call_args.kwargs["target_generation"] == 7

    def test_manager_error_propagates(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.swap_to_version.side_effect = RuntimeError("Index not found on disk")

        inp = SwapIndexInput(version="v-nonexistent", job_id="rebuild-1", target_generation=7)

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            with pytest.raises(RuntimeError, match="Index not found"):
                activity_env.run(swap_index_activity, inp)

    def test_swap_activates_generation_in_one_transaction(
        self, db_session: Session, activity_env: ActivityEnvironment, monkeypatch
    ) -> None:
        from activities.indexing.faiss_manager import FaissIndexManager
        from shared.models import JobType, SearchIndexState
        from tests.factories import create_index_build, create_job, create_search_index_state

        job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()
        create_index_build(session=db_session, version="v-old", is_active=True, dimension=768)
        create_index_build(session=db_session, version="v-new", is_active=False, dimension=768)
        create_search_index_state(
            session=db_session,
            desired_generation=5,
            active_generation=1,
            rebuild_job_id=job.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
        )
        db_session.flush()

        manager = FaissIndexManager()
        monkeypatch.setattr(manager, "_load_index_version", lambda version: None)

        inp = SwapIndexInput(version="v-new", job_id=job.id, target_generation=5)
        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=manager,
        ):
            activity_env.run(swap_index_activity, inp)

        active = {b.version: b.is_active for b in db_session.scalars(select(IndexBuild)).all()}
        assert active == {"v-old": False, "v-new": True}
        state = db_session.get(SearchIndexState, 1)
        assert state is not None and state.active_generation == 5


# ==========================================================================
# garbage_collect_indexes_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestGarbageCollectIndexesActivity:
    def test_returns_removed_versions(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.garbage_collect.return_value = ["v-old-001", "v-old-002"]

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            result = activity_env.run(garbage_collect_indexes_activity)

        assert result.removed_versions == ["v-old-001", "v-old-002"]

    def test_empty_result_when_nothing_to_gc(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.garbage_collect.return_value = []

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            result = activity_env.run(garbage_collect_indexes_activity)

        assert result.removed_versions == []
