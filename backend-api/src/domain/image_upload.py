from __future__ import annotations

import re
import uuid

from domain.image_ingest_input import StagedUploadInput
from shared.services.api_storage import ApiStorage

UPLOAD_STAGING_PREFIX = "uploads/staging"

_EXT_RE = re.compile(r"^[a-z0-9]{1,8}$")


def staging_key(filename: str | None, *, token: str | None = None) -> str:
    token = token or uuid.uuid4().hex
    ext = _extension(filename)
    suffix = f".{ext}" if ext else ""
    return f"{UPLOAD_STAGING_PREFIX}/{token}{suffix}"


def _extension(filename: str | None) -> str | None:
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext if _EXT_RE.match(ext) else None


class ImageUploadStager:
    def __init__(self, storage: ApiStorage) -> None:
        self._storage = storage

    async def stage(
        self,
        *,
        content: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> StagedUploadInput:
        key = staging_key(filename)
        await self._storage.upload_bytes(
            content, key, content_type=content_type or "application/octet-stream"
        )
        return StagedUploadInput(artifact_key=key)
