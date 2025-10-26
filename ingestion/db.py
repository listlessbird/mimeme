from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import CFG


PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=30000000000;", # ~30GB if OS allows; safe to ignore if not
    "PRAGMA page_size=32768;",
)

@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(CFG.db_path)
    conn.row_factory = sqlite3.Row
    try:
        for p in PRAGMAS:
            conn.execute(p)
        yield conn
        conn.commit()
    finally:
        conn.close()

def upsert_image(
    conn: sqlite3.Connection,
    rec: dict
) -> None:
    conn.execute(
        """
            INSERT INTO images (sha256, rel_path, width, height, format, phash)
            VALUES (:sha256, :rel_path, :width, :height, :format, :phash)
            ON CONFLICT(sha256) DO UPDATE SET
              rel_path=excluded.rel_path,
              width=CASE WHEN excluded.width IS NOT NULL THEN excluded.width ELSE width END,
              height=CASE WHEN excluded.height IS NOT NULL THEN excluded.height ELSE height END,
              format=CASE WHEN excluded.format IS NOT NULL THEN excluded.format ELSE format END,
              phash=CASE WHEN excluded.phash IS NOT NULL THEN excluded.phash ELSE phash END
              ;
        """,
        rec
    )

def bulk_upsert_images(
    conn: sqlite3.Connection,
    recs: list[dict]
) -> None:
    conn.executemany(
        """
        INSERT INTO images (sha256, rel_path, width, height, format, phash)
        VALUES (:sha256, :rel_path, :width, :height, :format, :phash)
        ON CONFLICT(sha256) DO UPDATE SET
        rel_path=excluded.rel_path,
        width=COALESCE(excluded.width, width),
        height=COALESCE(excluded.height, height),
        format=COALESCE(excluded.format, format),
        phash=COALESCE(excluded.phash, phash)
        ;
        """,
        recs
    )