from __future__ import annotations

from typing import BinaryIO

import pytest

from api.deps import get_storage
from shared.services.api_storage import AsyncApiStorage


class RecordingStorage:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        return f"https://fake/{key}?expires={expiration}"

    def upload_bytes(
        self, data: bytes | BinaryIO, key: str, content_type: str = "application/octet-stream"
    ) -> str:
        self.keys.add(key)
        return f"etag:{key}"

    def delete(self, key: str) -> None:
        self.keys.discard(key)

    def exists(self, key: str) -> bool:
        return key in self.keys


def test_get_storage_returns_async_api_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.deps.get_storage_service", RecordingStorage)

    storage = get_storage()

    assert isinstance(storage, AsyncApiStorage)
    assert storage.presign("images/source/example.jpg", expiration=42) == (
        "https://fake/images/source/example.jpg?expires=42"
    )
