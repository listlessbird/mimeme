"""Tests for the combined vision annotation activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from temporalio.testing import ActivityEnvironment

from activities.vision.activities import annotate_image_activity
from activities.vision.models import AnnotateImageInput, AnnotateImageOutput


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


class TestAnnotateImageActivity:
    async def test_returns_annotation_from_backend(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.annotate_image.return_value = AnnotateImageOutput(
            image_id=42,
            caption="A cat sitting on a keyboard",
            caption_model="moondream2",
            ocr_text="SHIP IT",
            ocr_model="moondream2",
        )

        inp = AnnotateImageInput(image_id=42, s3_key="images/test/cat.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(annotate_image_activity, inp)

        assert result.image_id == 42
        assert result.caption == "A cat sitting on a keyboard"
        assert result.caption_model == "moondream2"
        assert result.ocr_text == "SHIP IT"
        assert result.ocr_model == "moondream2"
        mock_backend.annotate_image.assert_called_once_with(inp)

    async def test_passes_length_parameter(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.annotate_image.return_value = AnnotateImageOutput(
            image_id=1,
            caption="Short",
            caption_model="moondream2",
            ocr_text="",
            ocr_model="moondream2",
        )

        inp = AnnotateImageInput(image_id=1, s3_key="images/test/img.jpg", length="short")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            await activity_env.run(annotate_image_activity, inp)

        called_input = mock_backend.annotate_image.call_args[0][0]
        assert called_input.length == "short"

    async def test_backend_error_propagates(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.annotate_image.side_effect = RuntimeError("CUDA out of memory")

        inp = AnnotateImageInput(image_id=1, s3_key="images/test/img.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                await activity_env.run(annotate_image_activity, inp)
