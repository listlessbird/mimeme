"""Tests for embed_batch_activity.

This activity delegates to the GPU backend to generate image and text
embeddings, then uploads them to S3.  Tests mock the backend and verify
correct delegation, partial failure handling, and output structure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from temporalio.testing import ActivityEnvironment

from activities.embedding.activity import embed_batch_activity
from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageInput,
    EmbedImageOutput,
)


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


class TestEmbedBatchActivity:
    async def test_successful_batch(self, activity_env: ActivityEnvironment) -> None:
        items = [
            EmbedImageInput(image_id=1, s3_key="images/test/1.jpg", sha256="hash1", dataset="test"),
            EmbedImageInput(image_id=2, s3_key="images/test/2.jpg", sha256="hash2", dataset="test"),
        ]
        expected_output = EmbedBatchOutput(
            results=[
                EmbedImageOutput(
                    image_id=1,
                    image_embedding_key="embeddings/1.npy",
                    text_embedding_key="embeddings/1_text.npy",
                    model="siglip2-base",
                    dimension=768,
                ),
                EmbedImageOutput(
                    image_id=2,
                    image_embedding_key="embeddings/2.npy",
                    text_embedding_key="embeddings/2_text.npy",
                    model="siglip2-base",
                    dimension=768,
                ),
            ],
            failed_ids=[],
        )

        mock_backend = AsyncMock()
        mock_backend.embed_batch.return_value = expected_output

        inp = EmbedBatchInput(items=items, dataset="test")

        with patch("activities.embedding.activity.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(embed_batch_activity, inp)

        assert len(result.results) == 2
        assert result.failed_ids == []
        assert result.results[0].image_id == 1
        assert result.results[0].dimension == 768
        mock_backend.embed_batch.assert_called_once_with(inp)

    async def test_partial_failure(self, activity_env: ActivityEnvironment) -> None:
        """When some images fail embedding, they appear in failed_ids."""
        mock_backend = AsyncMock()
        mock_backend.embed_batch.return_value = EmbedBatchOutput(
            results=[
                EmbedImageOutput(
                    image_id=1,
                    image_embedding_key="embeddings/1.npy",
                    text_embedding_key="",
                    model="siglip2-base",
                    dimension=768,
                ),
            ],
            failed_ids=[2, 3],
        )

        inp = EmbedBatchInput(
            items=[
                EmbedImageInput(image_id=1, s3_key="images/1.jpg", sha256="h1"),
                EmbedImageInput(image_id=2, s3_key="images/2.jpg", sha256="h2"),
                EmbedImageInput(image_id=3, s3_key="images/3.jpg", sha256="h3"),
            ],
        )

        with patch("activities.embedding.activity.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(embed_batch_activity, inp)

        assert len(result.results) == 1
        assert result.failed_ids == [2, 3]

    async def test_backend_error_propagates(self, activity_env: ActivityEnvironment) -> None:
        mock_backend = AsyncMock()
        mock_backend.embed_batch.side_effect = RuntimeError("GPU OOM")

        inp = EmbedBatchInput(
            items=[EmbedImageInput(image_id=1, s3_key="images/1.jpg", sha256="h1")],
        )

        with patch("activities.embedding.activity.get_gpu_backend", return_value=mock_backend):
            with pytest.raises(RuntimeError, match="GPU OOM"):
                await activity_env.run(embed_batch_activity, inp)

    async def test_empty_batch(self, activity_env: ActivityEnvironment) -> None:
        """Empty batch should still go through the backend cleanly."""
        mock_backend = AsyncMock()
        mock_backend.embed_batch.return_value = EmbedBatchOutput(results=[], failed_ids=[])

        inp = EmbedBatchInput(items=[])

        with patch("activities.embedding.activity.get_gpu_backend", return_value=mock_backend):
            result = await activity_env.run(embed_batch_activity, inp)

        assert result.results == []
        assert result.failed_ids == []
