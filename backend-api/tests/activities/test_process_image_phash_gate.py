"""Behavior spec for the pHash gate inside process_image_activity (slice 4).

Three ordered outcomes after a download lands a file:

1. **sha256 hit** — exact duplicate. Already worked; now it also records the
   reason (`SHA256`) and the canonical image it collapsed into.
2. **sha256 miss, pHash hit (<= 8)** — near-duplicate (a recompressed repost).
   Point at the canonical image, record `PHASH`, and *skip the upload* (no new
   bytes, no new Image row, no annotate/embed downstream).
3. **sha256 miss, pHash miss** — genuinely new. Upload, create the Image, and
   the new pHash becomes matchable for later runs.

The pHash index is injected by patching `get_phash_index` (same seam the tests
already use for `get_storage_service`). compute_sha256/compute_phash are patched
so we drive the gate deterministically without real image bytes.

Distances are against the all-zero pHash, so a hex with k bits set is exactly k
away (see test_phash_gate.py / test_phash_index.py).
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.activities.storage.activities import process_image_activity
from mimeme.activities.storage.models import ProcessImageInput, ProcessImageOutput
from mimeme.activities.storage.phash_index import PhashIndex
from mimeme.db.schema import DuplicateReason
from tests.factories import create_image

PHASH_ZERO = "0000000000000000"
PHASH_DISTANCE_8 = "00000000000000ff"  # near-duplicate of PHASH_ZERO
PHASH_FAR = "ffffffffffffffff"  # distance 64 — distinct


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


def _temp_image() -> str:
    """A throwaway file on disk; the activity unlinks it in its finally block."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        return f.name


def _run(
    env: ActivityEnvironment,
    inp: ProcessImageInput,
    *,
    index: PhashIndex,
    sha256: str,
    phash: str,
) -> tuple[ProcessImageOutput, MagicMock]:
    storage = MagicMock()
    storage.build_image_key.return_value = "images/test/new.jpg"
    storage.upload_file.return_value = "etag"
    with (
        patch(
            "mimeme.activities.storage.activities.get_media_storage_service", return_value=storage
        ),
        patch("mimeme.activities.storage.activities.get_phash_index", return_value=index),
        patch("mimeme.activities.storage.activities.compute_sha256", return_value=sha256),
        patch("mimeme.activities.storage.activities.compute_phash", return_value=phash),
        patch(
            "mimeme.activities.storage.activities.get_image_info", return_value=(800, 600, "jpeg")
        ),
    ):
        return env.run(process_image_activity, inp), storage


@pytest.mark.usefixtures("_patch_session_scope")
class TestProcessImagePhashGate:
    async def test_exact_duplicate_records_sha256_reason(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        canonical = create_image(session=db_session, sha256="exact-sha", phash=PHASH_ZERO)
        db_session.flush()

        inp = ProcessImageInput(
            local_path=_temp_image(), filename="dup.jpg", ingest_url_id=1, dataset="test"
        )
        result, storage = _run(
            activity_env, inp, index=PhashIndex(), sha256="exact-sha", phash=PHASH_ZERO
        )

        assert result.is_duplicate is True
        assert result.duplicate_reason == DuplicateReason.SHA256
        assert result.image_id == canonical.id
        assert result.duplicate_of_image_id == canonical.id
        storage.upload_file.assert_not_called()

    async def test_near_duplicate_is_deduped_via_phash(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        canonical = create_image(session=db_session, sha256="canonical-sha", phash=PHASH_ZERO)
        db_session.flush()

        index = PhashIndex()
        index.load_from_db(db_session)

        inp = ProcessImageInput(
            local_path=_temp_image(), filename="repost.jpg", ingest_url_id=2, dataset="test"
        )
        # sha256 misses (new bytes), pHash lands within 8 of the canonical.
        result, storage = _run(
            activity_env, inp, index=index, sha256="new-sha", phash=PHASH_DISTANCE_8
        )

        assert result.is_duplicate is True
        assert result.duplicate_reason == DuplicateReason.PHASH
        assert result.image_id == canonical.id
        assert result.duplicate_of_image_id == canonical.id
        storage.upload_file.assert_not_called()

    async def test_distinct_image_still_ingests(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        create_image(session=db_session, sha256="canonical-sha", phash=PHASH_ZERO)
        db_session.flush()

        index = PhashIndex()
        index.load_from_db(db_session)

        inp = ProcessImageInput(
            local_path=_temp_image(), filename="fresh.jpg", ingest_url_id=3, dataset="test"
        )
        result, storage = _run(activity_env, inp, index=index, sha256="fresh-sha", phash=PHASH_FAR)

        assert result.is_duplicate is False
        assert result.duplicate_reason is None
        assert result.duplicate_of_image_id is None
        storage.upload_file.assert_called_once()
