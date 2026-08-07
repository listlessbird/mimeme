from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

_MULTIPART_THRESHOLD = 8 * 1024 * 1024
_MULTIPART_CHUNK = 8 * 1024 * 1024
_PUT_BYTES_MAX = 5 * 1024 * 1024


class Error(Exception):
    pass


class Invalid(Error):
    pass


class Missing(Error):
    pass


class Denied(Error):
    pass


class Integrity(Error):
    pass


class Timeout(Error):
    pass


class Unavailable(Error):
    pass


class Counts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    get_object: int = 0
    head_object: int = 0
    head_bucket: int = 0
    put_object: int = 0
    create_multipart: int = 0
    upload_part: int = 0
    complete_multipart: int = 0
    list_page: int = 0

    @property
    def class_a(self) -> int:
        return (
            self.put_object
            + self.create_multipart
            + self.upload_part
            + self.complete_multipart
            + self.list_page
        )

    @property
    def class_b(self) -> int:
        return self.get_object + self.head_object + self.head_bucket


class Object(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str

    def __init__(self, key: str) -> None:
        super().__init__(key=key)

    @field_validator("key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("object key must be a non-empty trimmed string")
        if value.startswith("/") or value.endswith("/"):
            raise ValueError("object key must not start or end with '/'")
        if "//" in value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("object key must not contain empty or relative segments")
        return value

    def __str__(self) -> str:
        return self.key


class Checksum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        cleaned = value.lower()
        if len(cleaned) != 64 or any(c not in "0123456789abcdef" for c in cleaned):
            raise ValueError("sha256 checksum must be 64 hex characters")
        return cleaned

    @classmethod
    def of(cls, data: bytes) -> Checksum:
        return cls(value=hashlib.sha256(data).hexdigest())


class Info(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    object: Object
    length: int
    content_type: str | None = None
    checksum: Checksum | None = None
    etag: str | None = None
    modified_at: datetime | None = None


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_url: str
    region: str
    access_key: str
    secret_key: SecretStr
    bucket: str
    force_path_style: bool = True
    multipart_threshold: int = _MULTIPART_THRESHOLD
    multipart_chunk: int = _MULTIPART_CHUNK
    put_bytes_max: int = _PUT_BYTES_MAX
