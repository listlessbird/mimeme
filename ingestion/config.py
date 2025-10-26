from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    db_path: str = os.getenv("INGESTION_DB", "ingestion/data/db.sqlite3")
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    workers: int = int(os.getenv("INGESTION_WORKERS", "4"))
    batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "100"))

CFG = Config()