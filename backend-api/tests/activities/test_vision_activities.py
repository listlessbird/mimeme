"""Tests for vision activities: caption_activity, ocr_activity.

These activities delegate to a GPU backend (Moondream) via the GpuBackend
protocol.  Tests mock the backend and verify correct delegation, error
propagation, and output structure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from temporalio.testing import ActivityEnvironment

from activities.vision.activities import caption_activity, ocr_activity
from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ==========================================================================
# caption_activity
# ==========================================================================


class TestCaptionActivity:
    async def test_returns_caption_from_backend(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.caption.return_value = CaptionOutput(
            image_id=42,
            caption="A cat sitting on a keyboard",
            model="moondream2",
        )

        inp = CaptionInput(image_id=42, s3_key="images/test/cat.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(caption_activity, inp)

        assert result.image_id == 42
        assert result.caption == "A cat sitting on a keyboard"
        assert result.model == "moondream2"
        mock_backend.caption.assert_called_once_with(inp)

    async def test_passes_length_parameter(self, activity_env: ActivityEnvironment) -> None:
        """The length parameter should be forwarded to the backend."""
        mock_backend = AsyncMock()
        mock_backend.caption.return_value = CaptionOutput(
            image_id=1, caption="Short", model="moondream2"
        )

        inp = CaptionInput(image_id=1, s3_key="images/test/img.jpg", length="short")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            await activity_env.run(caption_activity, inp)

        called_input = mock_backend.caption.call_args[0][0]
        assert called_input.length == "short"

    async def test_backend_error_propagates(self, activity_env: ActivityEnvironment) -> None:
        """If the GPU backend raises, the error should propagate up."""
        mock_backend = AsyncMock()
        mock_backend.caption.side_effect = RuntimeError("CUDA out of memory")

        inp = CaptionInput(image_id=1, s3_key="images/test/img.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                await activity_env.run(caption_activity, inp)


# ==========================================================================
# ocr_activity
# ==========================================================================


class TestOCRActivity:
    async def test_returns_ocr_text_from_backend(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.ocr.return_value = OCROutput(
            image_id=7,
            text="WHEN YOU REALIZE\nIT'S MONDAY",
            model="moondream2",
        )

        inp = OCRInput(image_id=7, s3_key="images/test/meme.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(ocr_activity, inp)

        assert result.image_id == 7
        assert "MONDAY" in result.text
        assert result.model == "moondream2"
        mock_backend.ocr.assert_called_once_with(inp)

    async def test_empty_ocr_text(self, activity_env: ActivityEnvironment) -> None:
        """Images with no text should return empty string."""
        mock_backend = AsyncMock()
        mock_backend.ocr.return_value = OCROutput(
            image_id=8, text="", model="moondream2"
        )

        inp = OCRInput(image_id=8, s3_key="images/test/photo.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(ocr_activity, inp)

        assert result.text == ""

    async def test_backend_error_propagates(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.ocr.side_effect = ConnectionError("Backend unreachable")

        inp = OCRInput(image_id=1, s3_key="images/test/img.jpg")

        with patch("activities.vision.activities.get_gpu_backend", return_value=mock_backend):
            with pytest.raises(ConnectionError, match="Backend unreachable"):
                await activity_env.run(ocr_activity, inp)
