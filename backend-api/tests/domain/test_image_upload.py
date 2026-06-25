from __future__ import annotations

from unittest.mock import MagicMock

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
    def test_stores_bytes_and_returns_presigned_url(self) -> None:
        storage = MagicMock()
        storage.generate_presigned_url.return_value = "https://mock-s3/presigned"

        staged = ImageUploadStager(storage).stage(
            content=b"image-bytes", filename="meme.png", content_type="image/png"
        )

        storage.upload_bytes.assert_called_once()
        args, kwargs = storage.upload_bytes.call_args
        assert args[0] == b"image-bytes"
        assert args[1].startswith(f"{UPLOAD_STAGING_PREFIX}/")
        assert args[1].endswith(".png")
        assert kwargs["content_type"] == "image/png"

        assert staged.key == args[1]
        assert staged.url == "https://mock-s3/presigned"
        storage.generate_presigned_url.assert_called_once_with(
            staged.key, expiration=settings.s3_presigned_url_expiry
        )

    def test_defaults_content_type_when_missing(self) -> None:
        storage = MagicMock()
        storage.generate_presigned_url.return_value = "https://mock-s3/presigned"

        ImageUploadStager(storage).stage(content=b"x", filename=None, content_type=None)

        _, kwargs = storage.upload_bytes.call_args
        assert kwargs["content_type"] == "application/octet-stream"
