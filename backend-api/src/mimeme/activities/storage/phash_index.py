import threading

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import ORMImage
from mimeme.domain.phash_gate import find_near_duplicate, phash_to_uint64


class PhashIndex:
    def __init__(self) -> None:
        self._phashes = np.empty(0, dtype=np.uint64)
        self._image_ids = np.empty(0, dtype=np.int64)
        self._lock = threading.Lock()

    def add(self, image_id: int, phash: str | None) -> None:

        value = phash_to_uint64(phash)

        if value is None:
            return

        with self._lock:
            self._phashes = np.append(self._phashes, np.uint64(value))
            self._image_ids = np.append(self._image_ids, np.int64(image_id))

    def match(self, phash: str | None) -> int | None:
        if phash:
            value = phash_to_uint64(phash)
            return find_near_duplicate(value, self._phashes, self._image_ids)

        return None

    def load_from_db(self, session: Session) -> None:

        rows = session.execute(
            select(ORMImage.id, ORMImage.phash).where(ORMImage.phash.isnot(None))
        ).all()

        phashes: list[int] = []
        image_ids: list[int] = []

        for image_id, phash in rows:
            value = phash_to_uint64(phash)

            if value is None:
                continue

            phashes.append(value)
            image_ids.append(image_id)

        with self._lock:
            self._phashes = np.array(phashes, dtype=np.uint64)
            self._image_ids = np.array(image_ids, dtype=np.int64)


_index: PhashIndex | None = None


def get_phash_index() -> PhashIndex:
    global _index

    if _index is None:
        _index = PhashIndex()

    return _index


def reset_phash_index() -> None:
    global _index
    _index = None
