from __future__ import annotations

from typing import BinaryIO
from unittest.mock import AsyncMock, MagicMock

from domain.image_upload import UPLOAD_STAGING_PREFIX, ImageUploadStager, staging_key
from shared.config import settings


class TestStagingKey:
    def test_uses_lowercased_extension(self) -> None:
        assert staging_key("Meme.JPG", token="abc") == f"{UPLOAD_STAGING_PREFIX}/abc.jpg"

    def test_no_extension_when_missing(self) -> None:
        assert staging_key("noext", token="abc") == f"{UPLOAD_STAGING_PREFIX}/abc"

    def test_none_filename(self) -> None:
        assert staging_key(None, token="abc") == f"{UPLOAD_STAGING_PREFIX}/abc"

    def test_multi_dot_takes_last_segment(self) -> None:
        assert staging_key("archive.tar.gz", token="abc") == f"{UPLOAD_STAGING_PREFIX}/abc.gz"

    def test_rejects_non_alnum_extension(self) -> None:
        assert staging_key("bad.<x>", token="abc") == f"{UPLOAD_STAGING_PREFIX}/abc"

    def test_unique_tokens(self) -> None:
        assert staging_key("a.jpg") != staging_key("a.jpg")


class TestImageUploadStager:
    class FakeApiStorage:
        def __init__(self) -> None:
            self.uploaded: list[tuple[bytes | BinaryIO, str, str]] = []
            self.presigned: list[tuple[str, int]] = []

        def presign(self, key: str, expiration: int = 3600) -> str:
            self.presigned.append((key, expiration))
            return f"https://fake/{key}"

        async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
            self.uploaded.append((data, key, content_type))
            return f"etag:{key}"

        async def delete(self, key: str) -> None:
            pass

        async def exists(self, key: str) -> bool:
            return True

    async def test_stores_bytes_and_returns_presigned_url(self) -> None:
        storage = MagicMock()
        storage.upload_bytes = AsyncMock(return_value="etag")
        storage.presign.return_value = "https://mock-s3/presigned"

        staged = await ImageUploadStager(storage).stage(
            content=b"image-bytes", filename="meme.png", content_type="image/png"
        )

        storage.upload_bytes.assert_awaited_once()
        args, kwargs = storage.upload_bytes.await_args
        assert args[0] == b"image-bytes"
        assert args[1].startswith(f"{UPLOAD_STAGING_PREFIX}/")
        assert args[1].endswith(".png")
        assert kwargs["content_type"] == "image/png"

        assert staged.key == args[1]
        assert staged.url == "https://mock-s3/presigned"
        storage.presign.assert_called_once_with(
            staged.key, expiration=settings.s3_presigned_url_expiry
        )

    async def test_defaults_content_type_when_missing(self) -> None:
        storage = MagicMock()
        storage.upload_bytes = AsyncMock(return_value="etag")
        storage.presign.return_value = "https://mock-s3/presigned"

        await ImageUploadStager(storage).stage(content=b"x", filename=None, content_type=None)

        _, kwargs = storage.upload_bytes.await_args
        assert kwargs["content_type"] == "application/octet-stream"

    async def test_stage_uses_api_storage_presign_surface(self) -> None:
        storage = self.FakeApiStorage()

        staged = await ImageUploadStager(storage).stage(
            content=b"image-bytes", filename="meme.png", content_type="image/png"
        )

        assert storage.uploaded == [(b"image-bytes", staged.key, "image/png")]
        assert storage.presigned == [(staged.key, settings.s3_presigned_url_expiry)]
        assert staged.url == f"https://fake/{staged.key}"
