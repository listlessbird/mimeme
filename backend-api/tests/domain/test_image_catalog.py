from __future__ import annotations

from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from tests.factories import (
    create_annotation,
    create_artifact,
    create_image,
    create_processing,
)

from domain.image_catalog import ImageCatalog, ImageCatalogNotFoundError
from shared.models.orm import Annotation, Artifact, Image, Processing, ProcessingStatus

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


class FakeApiStorage:
    def __init__(self) -> None:
        self.presigned_keys: list[tuple[str, int]] = []

    def presign(self, key: str, expiration: int = 3600) -> str:
        self.presigned_keys.append((key, expiration))
        return f"https://fake/{key}"

    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    def delete(self, key: str) -> None:
        pass

    def exists(self, key: str) -> bool:
        return True


def test_list_images_empty(
    db_session: Session,
    mock_storage: MagicMock,
) -> None:
    page = ImageCatalog(mock_storage).list_images(limit=20, offset=0)

    assert page.images == []
    assert page.total == 0
    assert page.has_more is False


def test_list_images_with_pagination(db_session: Session, mock_storage: MagicMock) -> None:
    for _ in range(3):
        create_image(session=db_session)
    db_session.flush()

    page = ImageCatalog(mock_storage).list_images(limit=2, offset=0)

    assert len(page.images) == 2
    assert page.total == 3
    assert page.has_more is True


def test_list_images_filters_by_dataset(db_session: Session, mock_storage: MagicMock) -> None:
    create_image(session=db_session, dataset="cats")
    create_image(session=db_session, dataset="dogs")
    db_session.flush()

    page = ImageCatalog(mock_storage).list_images(
        limit=20,
        offset=0,
        dataset="cats",
    )

    assert page.total == 1
    assert page.images[0].dataset == "cats"


def test_list_images_sorts_oldest(db_session: Session, mock_storage: MagicMock) -> None:
    first = create_image(session=db_session)
    second = create_image(session=db_session)
    db_session.flush()

    page = ImageCatalog(mock_storage).list_images(
        limit=20,
        offset=0,
        sort="oldest",
    )

    assert [image.id for image in page.images] == [first.id, second.id]


@pytest.mark.parametrize(
    ("processing_overrides", "expected_status"),
    [
        ({}, "pending"),
        ({"embed_status": ProcessingStatus.DONE}, "done"),
        ({"ocr_status": ProcessingStatus.FAILED}, "failed"),
        ({"embed_status": ProcessingStatus.RUNNING}, "embedding"),
        ({"caption_status": ProcessingStatus.RUNNING}, "annotating"),
    ],
)
def test_list_images_projects_status(
    db_session: Session,
    mock_storage: MagicMock,
    processing_overrides: dict[str, ProcessingStatus],
    expected_status: str,
) -> None:
    image = create_image(session=db_session)
    create_processing(session=db_session, image=image, **processing_overrides)
    db_session.flush()

    page = ImageCatalog(mock_storage).list_images(limit=20, offset=0)

    assert page.images[0].status == expected_status


def test_list_images_filters_by_projected_status(
    db_session: Session,
    mock_storage: MagicMock,
) -> None:
    done = create_image(session=db_session)
    create_processing(session=db_session, image=done, embed_status=ProcessingStatus.DONE)
    failed = create_image(session=db_session)
    create_processing(session=db_session, image=failed, caption_status=ProcessingStatus.FAILED)
    db_session.flush()

    page = ImageCatalog(mock_storage).list_images(
        limit=20,
        offset=0,
        status="failed",
    )

    assert page.total == 1
    assert page.images[0].id == failed.id


def test_get_image_with_annotation(db_session: Session, mock_storage: MagicMock) -> None:
    image = create_image(session=db_session)
    create_annotation(session=db_session, image=image, caption_text="A cat", ocr_text="LOL")
    db_session.flush()

    result = ImageCatalog(mock_storage).get_image(image.id)

    assert result.id == image.id
    assert result.caption == "A cat"
    assert result.ocr_text == "LOL"


def test_get_image_without_annotation(db_session: Session, mock_storage: MagicMock) -> None:
    image = create_image(session=db_session)
    db_session.flush()

    result = ImageCatalog(mock_storage).get_image(image.id)

    assert result.id == image.id
    assert result.caption is None
    assert result.ocr_text is None


def test_get_image_missing(db_session: Session, mock_storage: MagicMock) -> None:
    with pytest.raises(ImageCatalogNotFoundError):
        ImageCatalog(mock_storage).get_image(999999)


def test_presigned_url_attached_when_storage_key_exists(
    db_session: Session,
    mock_storage: MagicMock,
) -> None:
    image = create_image(session=db_session, s3_key="images/test/example.jpg")
    db_session.flush()

    result = ImageCatalog(mock_storage).get_image(image.id)

    assert result.url == "https://mock-s3/presigned"
    mock_storage.presign.assert_called_once()


def test_image_catalog_uses_api_storage_presign_surface(db_session: Session) -> None:
    storage = FakeApiStorage()
    image = create_image(session=db_session, s3_key="images/test/example.jpg")
    db_session.flush()

    result = ImageCatalog(storage).get_image(image.id)

    assert result.url == "https://fake/images/test/example.jpg"
    assert storage.presigned_keys == [("images/test/example.jpg", 3600)]


def test_delete_image_removes_database_rows_and_storage_artifacts(
    db_session: Session,
    mock_storage: MagicMock,
) -> None:
    image = create_image(session=db_session, s3_key="images/test/example.jpg")
    create_processing(
        session=db_session,
        image=image,
        embed_s3_key="embeddings/test/example.npy",
    )
    create_annotation(session=db_session, image=image)
    create_artifact(session=db_session, image=image)
    db_session.flush()

    ImageCatalog(mock_storage).delete_image(image.id)

    assert db_session.get(Image, image.id) is None
    assert db_session.query(Processing).filter_by(image_id=image.id).first() is None
    assert db_session.query(Annotation).filter_by(image_id=image.id).first() is None
    assert db_session.query(Artifact).filter_by(image_id=image.id).first() is None
    assert mock_storage.delete.call_args_list[0].args == ("images/test/example.jpg",)
    assert mock_storage.delete.call_args_list[1].args == ("embeddings/test/example.npy",)
    assert mock_storage.delete.call_args_list[2].args == ("embeddings/test/example_text.npy",)


def test_delete_image_missing(db_session: Session, mock_storage: MagicMock) -> None:
    with pytest.raises(ImageCatalogNotFoundError):
        ImageCatalog(mock_storage).delete_image(999999)


def test_delete_image_storage_failure_does_not_block_database_deletion(
    db_session: Session,
    mock_storage: MagicMock,
) -> None:
    image = create_image(session=db_session, s3_key="images/test/example.jpg")
    db_session.flush()
    mock_storage.delete.side_effect = RuntimeError("storage unavailable")

    ImageCatalog(mock_storage).delete_image(image.id)

    assert db_session.get(Image, image.id) is None
