from __future__ import annotations

import numpy as np


def phash_to_uint64(hex_str: str | None) -> int | None:

    if not hex_str:
        return None

    try:
        return int(hex_str, 16)
    except (TypeError, ValueError):
        return None


def find_near_duplicate(
    new_phash: int | None, phashes: np.ndarray, image_ids: np.ndarray, threshold: int = 8
) -> int | None:

    if new_phash is None:
        return None

    if phashes.size == 0:
        return None

    # get the hamming distance between the hashes and the new hash
    # lower distance means its similar.

    distances = np.bitwise_count(phashes ^ np.uint64(new_phash))

    idx = int(np.argmin(distances))
    min_distance = int(distances[idx])

    if min_distance <= threshold:
        return int(image_ids[idx])

    return None
