from __future__ import annotations

from typing import Any, BinaryIO

import pytest
from botocore.client import ClientError

from shared.services.api_storage import AsyncApiStorage, BotoApiStorage


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


async def test_boto_api_storage_delegates_the_api_storage_surface() -> None:
    storage = RecordingStorage()
    api_storage = BotoApiStorage(storage)

    assert api_storage.presign("images/source/example.jpg", expiration=42) == (
        "https://fake/images/source/example.jpg?expires=42"
    )
    assert (
        await api_storage.upload_bytes(b"image", "uploads/staging/example.jpg", "image/jpeg")
        == "etag:uploads/staging/example.jpg"
    )
    assert await api_storage.exists("uploads/staging/example.jpg") is True

    await api_storage.delete("uploads/staging/example.jpg")

    assert await api_storage.exists("uploads/staging/example.jpg") is False
    assert storage.calls == [
        ("generate_presigned_url", "images/source/example.jpg", 42),
        ("upload_bytes", b"image", "uploads/staging/example.jpg", "image/jpeg"),
        ("exists", "uploads/staging/example.jpg"),
        ("delete", "uploads/staging/example.jpg"),
        ("exists", "uploads/staging/example.jpg"),
    ]


class FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.objects: set[str] = set()

    async def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            ("put_object", kwargs["Bucket"], kwargs["Key"], kwargs["Body"], kwargs["ContentType"])
        )
        self.objects.add(kwargs["Key"])
        return {"ETag": '"fake-etag"'}

    async def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete_object", kwargs["Bucket"], kwargs["Key"]))
        self.objects.discard(kwargs["Key"])
        return {}

    async def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_object", kwargs["Bucket"], kwargs["Key"]))
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    async def head_bucket(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_bucket", kwargs["Bucket"]))
        return {}


async def test_async_api_storage_uses_the_loop_client(monkeypatch: pytest.MonkeyPatch) -> None:
    presigner = RecordingStorage()
    client = FakeS3Client()

    async def _fake_get_client() -> FakeS3Client:
        return client

    monkeypatch.setattr("shared.services.api_storage.get_loop_client", _fake_get_client)
    monkeypatch.setattr(
        AsyncApiStorage, "bucket", property(lambda self: "test-bucket"), raising=True
    )

    api_storage = AsyncApiStorage(presigner)

    assert api_storage.presign("images/source/example.jpg", expiration=42) == (
        "https://fake/images/source/example.jpg?expires=42"
    )
    assert (
        await api_storage.upload_bytes(b"image", "uploads/staging/example.jpg", "image/jpeg")
        == "fake-etag"
    )
    assert await api_storage.exists("uploads/staging/example.jpg") is True

    await api_storage.delete("uploads/staging/example.jpg")

    assert await api_storage.exists("uploads/staging/example.jpg") is False
    assert await api_storage.bucket_exists() is True
    assert client.calls == [
        ("put_object", "test-bucket", "uploads/staging/example.jpg", b"image", "image/jpeg"),
        ("head_object", "test-bucket", "uploads/staging/example.jpg"),
        ("delete_object", "test-bucket", "uploads/staging/example.jpg"),
        ("head_object", "test-bucket", "uploads/staging/example.jpg"),
        ("head_bucket", "test-bucket"),
    ]
