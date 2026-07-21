"""Tests for storage activities: download_image, process_image, cleanup_temp_file.

process_image_activity creates Image and Processing DB records, so ORM cascade
and constraint tests live here alongside the activity tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.activities.storage.activities import (
    cleanup_temp_file_activity,
    download_image_activity,
    process_image_activity,
)
from mimeme.activities.storage.models import (
    DownloadImageInput,
    ProcessImageInput,
)
from mimeme.db.schema import (
    Annotation,
    Artifact,
    Image,
    Processing,
    ProcessingStatus,
)
from mimeme.domain.image_ingest_input import RemoteImageUrlInput, StagedUploadInput
from tests.factories import (
    create_annotation,
    create_artifact,
    create_image,
    create_processing,
)


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


# ==========================================================================
# download_image_activity
# ==========================================================================


class TestDownloadImageActivity:
    async def test_staged_upload_is_downloaded_through_artifact_storage(
        self, activity_env: ActivityEnvironment
    ) -> None:
        artifact_storage = MagicMock()
        artifact_storage.download_bytes.return_value = b"private-image"
        inp = DownloadImageInput(
            input=StagedUploadInput(artifact_key="uploads/staging/private.jpg"),
            job_id="test-job",
            ingest_url_id=10,
        )

        with patch(
            "mimeme.activities.storage.activities.get_artifact_storage_service",
            return_value=artifact_storage,
        ):
            result = activity_env.run(download_image_activity, inp)

        assert result.success is True
        assert Path(result.local_path).read_bytes() == b"private-image"
        artifact_storage.download_bytes.assert_called_once_with("uploads/staging/private.jpg")
        Path(result.local_path).unlink(missing_ok=True)

    async def test_successful_download(self, activity_env: ActivityEnvironment) -> None:
        """A successful HTTP download returns a local temp file path."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {"content-type": "image/jpeg"}
        fake_response.content = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        fake_response.raise_for_status = MagicMock()

        fake_client = MagicMock()
        fake_client.get.return_value = fake_response
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)

        inp = DownloadImageInput(
            input=RemoteImageUrlInput(url="https://example.com/cat.jpg"),
            job_id="test-job",
            ingest_url_id=1,
        )

        with patch("mimeme.activities.storage.activities.httpx.Client", return_value=fake_client):
            result = activity_env.run(download_image_activity, inp)

        assert result.success is True
        assert result.local_path != ""
        assert result.filename == "cat.jpg"
        # Clean up temp file
        Path(result.local_path).unlink(missing_ok=True)

    async def test_non_image_content_type_returns_failure(
        self, activity_env: ActivityEnvironment
    ) -> None:
        """If response content-type is not image/*, activity returns success=False."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {"content-type": "text/html"}
        fake_response.content = b"<html>Not an image</html>"
        fake_response.raise_for_status = MagicMock()

        fake_client = MagicMock()
        fake_client.get.return_value = fake_response
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)

        inp = DownloadImageInput(
            input=RemoteImageUrlInput(url="https://example.com/page.html"),
            job_id="test-job",
            ingest_url_id=2,
        )

        with patch("mimeme.activities.storage.activities.httpx.Client", return_value=fake_client):
            result = activity_env.run(download_image_activity, inp)

        assert result.success is False
        assert result.error is not None
        assert "content type" in result.error.lower() or "content_type" in result.error.lower()

    async def test_terminal_http_error_returns_failure(
        self, activity_env: ActivityEnvironment
    ) -> None:
        """Terminal HTTP errors are returned as success=False."""
        import httpx

        fake_response = MagicMock()
        fake_response.status_code = 404
        fake_response.headers = {"content-type": "text/html"}
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=fake_response
        )

        fake_client = MagicMock()
        fake_client.get.return_value = fake_response
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)

        inp = DownloadImageInput(
            input=RemoteImageUrlInput(url="https://example.com/missing.jpg"),
            job_id="test-job",
            ingest_url_id=3,
        )

        with patch("mimeme.activities.storage.activities.httpx.Client", return_value=fake_client):
            result = activity_env.run(download_image_activity, inp)

        assert result.success is False
        assert result.error is not None

    async def test_retryable_http_error_raises(self, activity_env: ActivityEnvironment) -> None:
        """Retryable HTTP errors are raised so Temporal can retry the activity."""
        import httpx

        fake_response = MagicMock()
        fake_response.status_code = 500
        fake_response.headers = {"content-type": "text/html"}
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=fake_response
        )

        fake_client = MagicMock()
        fake_client.get.return_value = fake_response
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)

        inp = DownloadImageInput(
            input=RemoteImageUrlInput(url="https://example.com/error.jpg"),
            job_id="test-job",
            ingest_url_id=4,
        )

        with (
            patch("mimeme.activities.storage.activities.httpx.Client", return_value=fake_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            activity_env.run(download_image_activity, inp)

    async def test_connection_error_raises_for_retry(
        self, activity_env: ActivityEnvironment
    ) -> None:
        """Network errors are raised so Temporal can retry the activity."""
        import httpx

        fake_client = MagicMock()
        fake_client.get.side_effect = httpx.ConnectError("Connection refused")
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)

        inp = DownloadImageInput(
            input=RemoteImageUrlInput(url="https://unreachable.example.com/img.jpg"),
            job_id="test-job",
            ingest_url_id=5,
        )

        with (
            patch("mimeme.activities.storage.activities.httpx.Client", return_value=fake_client),
            pytest.raises(httpx.ConnectError),
        ):
            activity_env.run(download_image_activity, inp)


# ==========================================================================
# process_image_activity
# ==========================================================================


@pytest.mark.usefixtures("_patch_session_scope")
class TestProcessImageActivity:
    async def test_new_image_creates_db_records(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """Processing a new image should create Image and Processing rows."""
        # Create a real temp image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Minimal JPEG-like content
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            temp_path = f.name

        mock_storage = MagicMock()
        mock_storage.build_image_key.return_value = "images/test/abc.jpg"
        mock_storage.upload_file.return_value = "mock-etag"

        inp = ProcessImageInput(
            local_path=temp_path,
            filename="cat.jpg",
            ingest_url_id=1,
            dataset="test-dataset",
        )

        with (
            patch(
                "mimeme.activities.storage.activities.get_media_storage_service",
                return_value=mock_storage,
            ),
            patch(
                "mimeme.activities.storage.activities.compute_sha256",
                return_value="unique-sha256-hash",
            ),
            patch("mimeme.activities.storage.activities.compute_phash", return_value="abcd1234"),
            patch(
                "mimeme.activities.storage.activities.get_image_info",
                return_value=(800, 600, "jpeg"),
            ),
        ):
            result = activity_env.run(process_image_activity, inp)

        assert result.is_duplicate is False
        assert result.sha256 == "unique-sha256-hash"
        assert result.s3_key == "images/test/abc.jpg"

        # Verify DB records were created
        img = db_session.query(Image).filter_by(id=result.image_id).first()
        assert img is not None
        assert img.sha256 == "unique-sha256-hash"
        assert img.dataset == "test-dataset"

        proc = db_session.query(Processing).filter_by(image_id=result.image_id).first()
        assert proc is not None

    async def test_duplicate_sha256_returns_existing(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        """If an image with the same SHA256 exists, return it as duplicate."""
        existing = create_image(
            session=db_session,
            sha256="existing-hash",
            s3_key="images/test/existing.jpg",
        )
        db_session.flush()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            temp_path = f.name

        mock_storage = MagicMock()

        inp = ProcessImageInput(
            local_path=temp_path,
            filename="cat.jpg",
            ingest_url_id=1,
            dataset="test-dataset",
        )

        with (
            patch(
                "mimeme.activities.storage.activities.get_media_storage_service",
                return_value=mock_storage,
            ),
            patch(
                "mimeme.activities.storage.activities.compute_sha256", return_value="existing-hash"
            ),
        ):
            result = activity_env.run(process_image_activity, inp)

        assert result.is_duplicate is True
        assert result.image_id == existing.id
        assert result.needs_annotation is True
        assert result.needs_embedding is True
        # Should NOT upload to S3 for duplicates
        mock_storage.upload_file.assert_not_called()

    async def test_fully_processed_duplicate_does_not_request_repair(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        existing = create_image(
            session=db_session,
            sha256="processed-hash",
            s3_key="images/test/processed.jpg",
        )
        create_processing(
            session=db_session,
            image=existing,
            caption_status=ProcessingStatus.DONE,
            ocr_status=ProcessingStatus.DONE,
            embed_status=ProcessingStatus.DONE,
            embed_s3_key="embeddings/test/processed.npy",
        )
        create_annotation(
            session=db_session,
            image=existing,
            caption_text="existing caption",
            ocr_text="existing ocr",
        )
        db_session.flush()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            temp_path = f.name

        mock_storage = MagicMock()

        inp = ProcessImageInput(
            local_path=temp_path,
            filename="cat.jpg",
            ingest_url_id=1,
            dataset="test-dataset",
        )

        with (
            patch(
                "mimeme.activities.storage.activities.get_media_storage_service",
                return_value=mock_storage,
            ),
            patch(
                "mimeme.activities.storage.activities.compute_sha256", return_value="processed-hash"
            ),
        ):
            result = activity_env.run(process_image_activity, inp)

        assert result.is_duplicate is True
        assert result.image_id == existing.id
        assert result.needs_annotation is False
        assert result.needs_embedding is False
        assert result.existing_caption == "existing caption"
        assert result.existing_ocr_text == "existing ocr"
        mock_storage.upload_file.assert_not_called()


# ==========================================================================
# cleanup_temp_file_activity
# ==========================================================================


class TestCleanupTempFileActivity:
    async def test_deletes_existing_file(self, activity_env: ActivityEnvironment) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        assert Path(temp_path).exists()
        activity_env.run(cleanup_temp_file_activity, temp_path)
        assert not Path(temp_path).exists()

    async def test_missing_file_does_not_raise(self, activity_env: ActivityEnvironment) -> None:
        """Cleaning up a nonexistent file should not raise."""
        activity_env.run(cleanup_temp_file_activity, "/tmp/nonexistent-file-12345.jpg")


# ==========================================================================
# Image ORM cascade deletes — process_image creates these records, so cascade
# correctness is critical for this activity's data integrity
# ==========================================================================


class TestImageCascadeDelete:
    def test_delete_image_cascades_to_processing(self, db_session: Session) -> None:
        image = create_image(session=db_session)
        create_processing(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Processing).filter_by(image_id=image_id).first() is None

    def test_delete_image_cascades_to_annotation(self, db_session: Session) -> None:
        image = create_image(session=db_session)
        create_annotation(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Annotation).filter_by(image_id=image_id).first() is None

    def test_delete_image_cascades_to_artifacts(self, db_session: Session) -> None:
        image = create_image(session=db_session)
        create_artifact(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Artifact).filter_by(image_id=image_id).first() is None

    def test_delete_image_without_related_records(self, db_session: Session) -> None:
        image = create_image(session=db_session)
        db_session.flush()

        db_session.delete(image)
        db_session.flush()
        assert db_session.query(Image).filter_by(id=image.id).first() is None


class TestImageUniqueConstraints:
    def test_sha256_unique_constraint(self, db_session: Session) -> None:
        create_image(session=db_session, sha256="abc123")
        db_session.flush()

        img2 = Image(sha256="abc123", dataset="other")
        db_session.add(img2)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_different_sha256_allowed(self, db_session: Session) -> None:
        img1 = create_image(session=db_session, sha256="hash1")
        img2 = create_image(session=db_session, sha256="hash2")
        db_session.flush()
        assert img1.id != img2.id
