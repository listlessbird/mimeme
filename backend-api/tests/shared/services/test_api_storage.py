from __future__ import annotations

from typing import BinaryIO

from shared.services.api_storage import BotoApiStorage


class RecordingStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.keys: set[str] = set()

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        self.calls.append(("generate_presigned_url", key, expiration))
        return f"https://fake/{key}?expires={expiration}"

    def upload_bytes(
        self, data: bytes | BinaryIO, key: str, content_type: str = "application/octet-stream"
    ) -> str:
        self.calls.append(("upload_bytes", data, key, content_type))
        self.keys.add(key)
        return f"etag:{key}"

    def delete(self, key: str) -> None:
        self.calls.append(("delete", key))
        self.keys.discard(key)

    def exists(self, key: str) -> bool:
        self.calls.append(("exists", key))
        return key in self.keys


def test_boto_api_storage_delegates_the_api_storage_surface() -> None:
    storage = RecordingStorage()
    api_storage = BotoApiStorage(storage)

    assert api_storage.presign("images/source/example.jpg", expiration=42) == (
        "https://fake/images/source/example.jpg?expires=42"
    )
    assert (
        api_storage.upload_bytes(b"image", "uploads/staging/example.jpg", "image/jpeg")
        == "etag:uploads/staging/example.jpg"
    )
    assert api_storage.exists("uploads/staging/example.jpg") is True

    api_storage.delete("uploads/staging/example.jpg")

    assert api_storage.exists("uploads/staging/example.jpg") is False
    assert storage.calls == [
        ("generate_presigned_url", "images/source/example.jpg", 42),
        ("upload_bytes", b"image", "uploads/staging/example.jpg", "image/jpeg"),
        ("exists", "uploads/staging/example.jpg"),
        ("delete", "uploads/staging/example.jpg"),
        ("exists", "uploads/staging/example.jpg"),
    ]
