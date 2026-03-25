"""Tests for indexing activities: build_index, swap_index, garbage_collect.

These activities interact with FaissIndexManager, S3, and DB.  Tests mock the
heavy dependencies and verify correct orchestration, DB state, and error handling.

Note: These are sync activities (not async), so activity_env.run() returns
the result directly without needing await.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from activities.indexing.activities import (
    build_index_activity,
    garbage_collect_indexes_activity,
    swap_index_activity,
)
from activities.indexing.models import BuildIndexInput, SwapIndexInput
from shared.models.orm import ProcessingStatus
from tests.factories import create_image, create_processing


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ==========================================================================
# build_index_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestBuildIndexActivity:
    def test_builds_index_from_embeddings(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """Happy path: done embeddings exist, index is built and returned."""
        for i in range(3):
            image = create_image(session=db_session)
            proc = create_processing(session=db_session, image=image)
            proc.embed_status = ProcessingStatus.DONE
            proc.embed_s3_key = f"embeddings/{image.id}.npy"
        db_session.flush()

        mock_storage = MagicMock()
        mock_storage.download_numpy.return_value = np.random.rand(768).astype(np.float32)
        mock_storage.build_index_key.return_value = "indexes/v-test/index.faiss"

        mock_build_result = MagicMock()
        mock_build_result.version = "v-test-001"
        mock_build_result.text_num_vectors = None
        mock_build_result.text_s3_key = None

        mock_manager = MagicMock()
        mock_manager.build_index.return_value = mock_build_result

        inp = BuildIndexInput(model_name="siglip2-base", index_type="flat")

        with (
            patch("activities.indexing.activities.get_storage_service", return_value=mock_storage),
            patch(
                "activities.indexing.activities.FaissIndexManager.get_instance",
                return_value=mock_manager,
            ),
        ):
            result = activity_env.run(build_index_activity, inp)

        assert result.version == "v-test-001"
        assert result.num_vectors == 3
        assert result.dimension == 768
        mock_manager.build_index.assert_called_once()

    def test_no_embeddings_raises(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """If no embeddings exist, the activity should raise ValueError."""
        mock_storage = MagicMock()
        mock_manager = MagicMock()

        inp = BuildIndexInput(model_name="siglip2-base")

        with (
            patch("activities.indexing.activities.get_storage_service", return_value=mock_storage),
            patch(
                "activities.indexing.activities.FaissIndexManager.get_instance",
                return_value=mock_manager,
            ),
        ):
            with pytest.raises(ValueError, match="No embeddings"):
                activity_env.run(build_index_activity, inp)

    def test_missing_s3_vectors_are_skipped(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """If some embeddings fail to download from S3, they are skipped."""
        from botocore.exceptions import ClientError

        for i in range(3):
            image = create_image(session=db_session)
            proc = create_processing(session=db_session, image=image)
            proc.embed_status = ProcessingStatus.DONE
            proc.embed_s3_key = f"embeddings/{image.id}.npy"
        db_session.flush()

        call_count = 0

        def _download_numpy(key: str) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClientError({"Error": {"Code": "404"}}, "GetObject")
            return np.random.rand(768).astype(np.float32)

        mock_storage = MagicMock()
        mock_storage.download_numpy.side_effect = _download_numpy
        mock_storage.build_index_key.return_value = "indexes/v-test/index.faiss"

        mock_build_result = MagicMock()
        mock_build_result.version = "v-test-002"
        mock_build_result.text_num_vectors = None
        mock_build_result.text_s3_key = None

        mock_manager = MagicMock()
        mock_manager.build_index.return_value = mock_build_result

        inp = BuildIndexInput(model_name="siglip2-base")

        with (
            patch("activities.indexing.activities.get_storage_service", return_value=mock_storage),
            patch(
                "activities.indexing.activities.FaissIndexManager.get_instance",
                return_value=mock_manager,
            ),
        ):
            result = activity_env.run(build_index_activity, inp)

        assert result.num_vectors == 2  # 1 skipped


# ==========================================================================
# swap_index_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestSwapIndexActivity:
    def test_delegates_to_manager(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()

        inp = SwapIndexInput(version="v-test-001")

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            activity_env.run(swap_index_activity, inp)

        mock_manager.swap_to_version.assert_called_once()
        call_args = mock_manager.swap_to_version.call_args
        assert call_args[0][0] == "v-test-001"

    def test_manager_error_propagates(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.swap_to_version.side_effect = RuntimeError("Index not found on disk")

        inp = SwapIndexInput(version="v-nonexistent")

        with patch(
            "activities.indexing.activities.FaissIndexManager.get_instance",
            return_value=mock_manager,
        ):
            with pytest.raises(RuntimeError, match="Index not found"):
                activity_env.run(swap_index_activity, inp)


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
