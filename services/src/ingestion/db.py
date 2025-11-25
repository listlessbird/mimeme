from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def connect():
    raise NotImplementedError("Deprecated. Use session_scope from ingestion.database")


def upsert_image(*args, **kwargs):
    raise NotImplementedError("Deprecated. Use repositories/images.bulk_upsert")


def bulk_upsert_images(*args, **kwargs):
    raise NotImplementedError("Deprecated. Use repositories/images.bulk_upsert")
