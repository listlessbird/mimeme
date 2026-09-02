from __future__ import annotations

import hashlib
import re
import uuid

TASK_QUEUE = "mimeme-v2"
WORKFLOW = "mimeme.ingest.v2"
ITEM_ACTIVITY = "mimeme.ingest.item.v2"
BATCH_ACTIVITY = "mimeme.ingest.batch.v2"
FINISH_ACTIVITY = "mimeme.ingest.finish.v2"

# One inference child and one image child serve the whole compute container. Keep
# only a few ingestion batches in flight so downloads overlap without flooding either.
FANOUT = 2
EMBED_BATCH_SIZE = 16

MAX_IMAGE_BYTES = 64 * 1024 * 1024
INGEST_STAGING_PREFIX = "uploads/ingest-staging"
UPLOAD_STAGING_PREFIX = "uploads/staging"
ERROR_LIMIT = 500

IMAGE_ACCEPT_HEADER = "image/avif,image/webp,image/apng,image/png,image/jpeg,image/*,*/*;q=0.8"
DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)
_RETRYABLE_4XX = frozenset({408, 425, 429})

# Fixed transaction-level advisory lock guarding the SHA duplicate decision and
# canonical image insert. All ingest workers contend on this single key.
DEDUP_LOCK_KEY = 0x6D696E67  # "ming"


def workflow_id(job_id: str) -> str:
    return f"ingest-v2-{job_id}"


def staging_key(item_id: int) -> str:
    return f"{INGEST_STAGING_PREFIX}/{item_id}"


_EXT_RE = re.compile(r"^[a-z0-9]{1,8}$")


def upload_staging_key(filename: str | None, *, token: str | None = None) -> str:
    token = token or uuid.uuid4().hex
    ext = _extension(filename)
    suffix = f".{ext}" if ext else ""
    return f"{UPLOAD_STAGING_PREFIX}/{token}{suffix}"


def _extension(filename: str | None) -> str | None:
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext if _EXT_RE.match(ext) else None


def canonical_media_key(*, sha256: str, dataset: str | None, image_format: str | None) -> str:
    ext = (image_format or "jpg").lower().lstrip(".")
    source = dataset if dataset else "api-ingested"
    return f"images/{source}/{sha256[:2]}/{sha256[2:4]}/{sha256}.{ext}"


def content_type_for(image_format: str | None) -> str:
    fmt = (image_format or "").lower()
    if fmt in {"jpg", "jpeg"}:
        return "image/jpeg"
    if fmt:
        return f"image/{fmt}"
    return "application/octet-stream"


def is_terminal_http_status(status: int) -> bool:
    return 400 <= status < 500 and status not in _RETRYABLE_4XX


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def truncate_error(message: str, limit: int = ERROR_LIMIT) -> str:
    cleaned = message.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def progress_percent(completed: int, total: int) -> float:
    return (completed / total) * 100 if total > 0 else 0.0
