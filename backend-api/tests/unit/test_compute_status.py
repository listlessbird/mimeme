"""Tests for the _compute_status helper in the images router."""

from __future__ import annotations

from unittest.mock import MagicMock

from api.models.images import ImageStatus
from api.routers.images import _compute_status
from shared.models.orm import ProcessingStatus


def _make_proc(**overrides: object) -> MagicMock:
    proc = MagicMock()
    proc.ocr_status = overrides.get("ocr_status", ProcessingStatus.PENDING)
    proc.caption_status = overrides.get("caption_status", ProcessingStatus.PENDING)
    proc.embed_status = overrides.get("embed_status", ProcessingStatus.PENDING)
    return proc


class TestComputeStatus:
    def test_no_processing_record_returns_pending(self) -> None:
        assert _compute_status(None) == ImageStatus.PENDING

    def test_all_pending_returns_pending(self) -> None:
        proc = _make_proc()
        assert _compute_status(proc) == ImageStatus.PENDING

    def test_embed_done_returns_done(self) -> None:
        proc = _make_proc(embed_status=ProcessingStatus.DONE)
        assert _compute_status(proc) == ImageStatus.DONE

    def test_ocr_failed_returns_failed(self) -> None:
        proc = _make_proc(ocr_status=ProcessingStatus.FAILED)
        assert _compute_status(proc) == ImageStatus.FAILED

    def test_caption_failed_returns_failed(self) -> None:
        proc = _make_proc(caption_status=ProcessingStatus.FAILED)
        assert _compute_status(proc) == ImageStatus.FAILED

    def test_embed_failed_returns_failed(self) -> None:
        proc = _make_proc(embed_status=ProcessingStatus.FAILED)
        assert _compute_status(proc) == ImageStatus.FAILED

    def test_embed_running_returns_embedding(self) -> None:
        proc = _make_proc(embed_status=ProcessingStatus.RUNNING)
        assert _compute_status(proc) == ImageStatus.EMBEDDING

    def test_caption_running_returns_annotating(self) -> None:
        proc = _make_proc(caption_status=ProcessingStatus.RUNNING)
        assert _compute_status(proc) == ImageStatus.ANNOTATING

    def test_ocr_running_returns_annotating(self) -> None:
        proc = _make_proc(ocr_status=ProcessingStatus.RUNNING)
        assert _compute_status(proc) == ImageStatus.ANNOTATING

    def test_failed_takes_priority_over_running(self) -> None:
        """If embed is running but OCR failed, status should be FAILED."""
        proc = _make_proc(
            ocr_status=ProcessingStatus.FAILED,
            embed_status=ProcessingStatus.RUNNING,
        )
        assert _compute_status(proc) == ImageStatus.FAILED

    def test_done_takes_priority_over_other_statuses(self) -> None:
        """If embed is DONE, overall is DONE regardless of other stages."""
        proc = _make_proc(
            ocr_status=ProcessingStatus.RUNNING,
            caption_status=ProcessingStatus.PENDING,
            embed_status=ProcessingStatus.DONE,
        )
        assert _compute_status(proc) == ImageStatus.DONE
