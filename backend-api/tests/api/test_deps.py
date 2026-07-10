from __future__ import annotations

from typing import BinaryIO

import pytest

from api.deps import get_storage


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


async def test_get_storage_returns_api_storage_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.deps.get_storage_service", RecordingStorage)

    storage = get_storage()

    assert storage.presign("images/source/example.jpg", expiration=42) == (
        "https://fake/images/source/example.jpg?expires=42"
    )
    assert await storage.upload_bytes(b"image", "uploads/staging/example.jpg", "image/jpeg") == (
        "etag:uploads/staging/example.jpg"
    )
    assert await storage.exists("uploads/staging/example.jpg") is True

    await storage.delete("uploads/staging/example.jpg")

    assert await storage.exists("uploads/staging/example.jpg") is False
