from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

# Project root is two levels up from this file: src/ingestion/config.py -> ingestion/
_PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass(frozen=True)
class Config:
    db_path: str = os.getenv("INGESTION_DB", str(_PROJECT_ROOT / "data" / "db.sqlite3"))
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    workers: int = int(os.getenv("INGESTION_WORKERS", "4"))
    batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "100"))
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")
    s3_region: str = os.getenv("S3_REGION", "auto")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_access_key_id: str = os.getenv("S3_ACCESS_KEY_ID", "")
    s3_secret_access_key: str = os.getenv("S3_SECRET_ACCESS_KEY", "")
    s3_force_path_style: bool = os.getenv("S3_FORCE_PATH_STYLE", "true") == "true"
    s3_prefix: str = os.getenv("S3_PREFIX", "memes")

CFG = Config()