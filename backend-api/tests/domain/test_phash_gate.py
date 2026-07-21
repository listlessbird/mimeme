"""Unit tests for the pure pHash near-duplicate matcher.

This is the high-value pure decision module for issue 01: given a new pHash, an
array of existing pHashes, and a threshold, decide whether the new image is a
near-duplicate of an existing one. No I/O.

The contract these tests pin down:

    find_near_duplicate(
        new_phash: int | None,        # the candidate image's 64-bit pHash
        phashes: np.ndarray,          # uint64, the known pHashes
        image_ids: np.ndarray,        # int64, parallel to `phashes`
        threshold: int = 8,
    ) -> int | None                   # matched image_id, or None to ingest

    phash_to_uint64(hex_str: str | None) -> int | None

Distances are Hamming distances over the 64 bits. Because every test pHash is
measured against 0x0, "a pHash with k bits set" sits at exactly distance k from
the candidate 0x0 — that is how the boundary tests below are constructed.

Bias-to-ingest is the load-bearing rule: when we cannot be confident (no
candidate pHash, empty index, nothing within threshold) the answer is None
("ingest"), never a guessed match — two distinct memes must never collapse.

Prior art: tests/domain/test_ingest_policy.py.
"""

from __future__ import annotations

import numpy as np

from mimeme.domain.phash_gate import find_near_duplicate, phash_to_uint64

# Candidate every case compares against. Distance to any other value is just the
# popcount (number of set bits) of that other value.
CANDIDATE = np.uint64(0x0)

# Values whose set-bit counts give exact, known Hamming distances from CANDIDATE.
DISTANCE_0 = 0x0  # 0 bits set
DISTANCE_7 = 0x7F  # 0b111_1111  -> 7 bits set
DISTANCE_8 = 0xFF  # 0b1111_1111 -> 8 bits set
DISTANCE_9 = 0x1FF  # 0b1_1111_1111 -> 9 bits set


def _index(*pairs: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build (phashes, image_ids) arrays from (phash, image_id) pairs."""
    phashes = np.array([p for p, _ in pairs], dtype=np.uint64)
    image_ids = np.array([i for _, i in pairs], dtype=np.int64)
    return phashes, image_ids


# ---------------------------------------------------------------------------
# find_near_duplicate — matching
# ---------------------------------------------------------------------------


def test_exact_match_returns_its_image_id() -> None:
    phashes, image_ids = _index((0xFFFFFFFFFFFFFFFF, 10), (DISTANCE_0, 20))

    assert find_near_duplicate(0x0, phashes, image_ids) == 20


def test_distance_just_below_threshold_matches() -> None:
    phashes, image_ids = _index((DISTANCE_7, 42))

    assert find_near_duplicate(0x0, phashes, image_ids) == 42


def test_distance_at_threshold_matches() -> None:
    """Distance 8 is inclusive: `<= 8` is a near-duplicate."""
    phashes, image_ids = _index((DISTANCE_8, 42))

    assert find_near_duplicate(0x0, phashes, image_ids) == 42


def test_distance_above_threshold_does_not_match() -> None:
    """Distance 9 is over the line — ingest as a distinct image."""
    phashes, image_ids = _index((DISTANCE_9, 42))

    assert find_near_duplicate(0x0, phashes, image_ids) is None


def test_returns_nearest_when_multiple_candidates() -> None:
    """Among several within-or-near matches, the closest wins."""
    phashes, image_ids = _index((DISTANCE_8, 1), (DISTANCE_7, 2), (DISTANCE_0, 3))

    assert find_near_duplicate(0x0, phashes, image_ids) == 3


def test_custom_threshold_is_respected() -> None:
    phashes, image_ids = _index((DISTANCE_8, 42))

    assert find_near_duplicate(0x0, phashes, image_ids, threshold=7) is None
    assert find_near_duplicate(0x0, phashes, image_ids, threshold=8) == 42


# ---------------------------------------------------------------------------
# find_near_duplicate — bias-to-ingest
# ---------------------------------------------------------------------------


def test_empty_index_biases_to_ingest() -> None:
    phashes, image_ids = _index()

    assert find_near_duplicate(0x0, phashes, image_ids) is None


def test_missing_candidate_phash_biases_to_ingest() -> None:
    """A candidate with no computable pHash is never collapsed into a match."""
    phashes, image_ids = _index((DISTANCE_0, 20))

    assert find_near_duplicate(None, phashes, image_ids) is None


# ---------------------------------------------------------------------------
# phash_to_uint64
# ---------------------------------------------------------------------------


def test_phash_to_uint64_parses_16_char_hex() -> None:
    # imagehash stringifies a 64-bit hash as 16 hex chars.
    assert phash_to_uint64("00000000000000ff") == 255


def test_phash_to_uint64_handles_full_width_value() -> None:
    assert phash_to_uint64("ffffffffffffffff") == 0xFFFFFFFFFFFFFFFF


def test_phash_to_uint64_returns_none_for_missing_value() -> None:
    assert phash_to_uint64(None) is None
    assert phash_to_uint64("") is None


def test_phash_to_uint64_returns_none_for_unparseable_value() -> None:
    assert phash_to_uint64("not-hex") is None
