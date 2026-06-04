"""Spec for the worker-startup pHash index (issue 01, slice 3).

The index is the thin infra adapter around the pure matcher in
`domain.phash_gate`. It holds every known pHash in memory (a uint64 numpy
array), is loaded once from the DB at worker startup, and is appended to as new
images are stored.

Interface decision pinned here: the index speaks **hex strings** — the same
currency the `images.phash` column and `compute_phash` produce — and converts
to uint64 internally. Callers never touch int conversion.

    index.load_from_db(session)          # build arrays from images.phash
    index.match(phash_hex) -> int | None # canonical image_id, or None to ingest
    index.add(image_id, phash_hex)       # remember a newly-stored image

    get_phash_index() -> PhashIndex      # process singleton
    reset_phash_index()                  # drop the singleton (tests/worker reload)

Distances are taken against the all-zero pHash, so "a hex with k bits set" sits
at exactly Hamming distance k — the same trick as test_phash_gate.py.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from activities.storage.phash_index import (
    PhashIndex,
    get_phash_index,
    reset_phash_index,
)
from tests.factories import create_image

PHASH_ZERO = "0000000000000000"  # distance 0
PHASH_DISTANCE_8 = "00000000000000ff"  # 8 bits set -> distance 8 from PHASH_ZERO
PHASH_DISTANCE_9 = "00000000000001ff"  # 9 bits set -> distance 9 from PHASH_ZERO
PHASH_DISTANCE_4 = "000000000000000f"  # 4 bits set -> distance 4 from PHASH_ZERO


@pytest.fixture(autouse=True)
def _fresh_singleton() -> None:
    """Each test starts with no process-global index."""
    reset_phash_index()


# ---------------------------------------------------------------------------
# load_from_db + match
# ---------------------------------------------------------------------------


def test_load_then_match_finds_near_duplicate(db_session: Session) -> None:
    canonical = create_image(session=db_session, phash=PHASH_ZERO)
    db_session.flush()

    index = PhashIndex()
    index.load_from_db(db_session)

    assert index.match(PHASH_DISTANCE_8) == canonical.id


def test_match_returns_none_when_nothing_within_threshold(db_session: Session) -> None:
    create_image(session=db_session, phash=PHASH_ZERO)
    db_session.flush()

    index = PhashIndex()
    index.load_from_db(db_session)

    assert index.match(PHASH_DISTANCE_9) is None


def test_match_on_empty_index_biases_to_ingest() -> None:
    index = PhashIndex()

    assert index.match(PHASH_ZERO) is None


def test_load_skips_images_without_a_phash(db_session: Session) -> None:
    create_image(session=db_session, phash=None)
    db_session.flush()

    index = PhashIndex()
    index.load_from_db(db_session)  # must not raise on the null pHash

    assert index.match(PHASH_ZERO) is None


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_added_phash_becomes_matchable(db_session: Session) -> None:
    index = PhashIndex()
    index.add(image_id=99, phash=PHASH_ZERO)

    assert index.match(PHASH_DISTANCE_4) == 99


def test_adding_a_null_phash_is_ignored(db_session: Session) -> None:
    index = PhashIndex()
    index.add(image_id=99, phash=None)

    assert index.match(PHASH_ZERO) is None


# ---------------------------------------------------------------------------
# singleton accessor
# ---------------------------------------------------------------------------


def test_get_phash_index_returns_one_shared_instance() -> None:
    assert get_phash_index() is get_phash_index()


def test_reset_replaces_the_singleton() -> None:
    first = get_phash_index()
    reset_phash_index()

    assert get_phash_index() is not first
