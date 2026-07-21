from __future__ import annotations

from typing import Any

import pytest
from botocore.client import ClientError

from mimeme.shared.services.api_storage import AsyncApiStorage
from mimeme.shared.services.storage import S3Config


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
    client = FakeS3Client()

    async def _fake_get_client(config: S3Config) -> FakeS3Client:
        return client

    monkeypatch.setattr("mimeme.shared.services.api_storage.get_loop_client", _fake_get_client)
    api_storage = AsyncApiStorage(
        S3Config(
            endpoint_url="http://example.test",
            region="auto",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket",
            force_path_style=True,
        )
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
