"""Tests for Image model cascade deletes and unique constraints."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from shared.models.orm import Annotation, Artifact, Image, Processing
from tests.factories import (
    AnnotationFactory,
    ArtifactFactory,
    ImageFactory,
    ProcessingFactory,
)


class TestImageCascadeDelete:
    def test_delete_image_cascades_to_processing(self, db_session) -> None:
        image = ImageFactory(session=db_session)
        ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Processing).filter_by(image_id=image_id).first() is None

    def test_delete_image_cascades_to_annotation(self, db_session) -> None:
        image = ImageFactory(session=db_session)
        AnnotationFactory(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Annotation).filter_by(image_id=image_id).first() is None

    def test_delete_image_cascades_to_artifacts(self, db_session) -> None:
        image = ImageFactory(session=db_session)
        ArtifactFactory(session=db_session, image=image)
        db_session.flush()

        image_id = image.id
        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Artifact).filter_by(image_id=image_id).first() is None

    def test_delete_image_without_related_records(self, db_session) -> None:
        """Deleting an image with no Processing/Annotation should not error."""
        image = ImageFactory(session=db_session)
        db_session.flush()

        db_session.delete(image)
        db_session.flush()

        assert db_session.query(Image).filter_by(id=image.id).first() is None


class TestImageUniqueConstraints:
    def test_sha256_unique_constraint(self, db_session) -> None:
        img1 = ImageFactory(session=db_session, sha256="abc123")
        db_session.flush()

        img2 = Image(sha256="abc123", dataset="other")
        db_session.add(img2)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_different_sha256_allowed(self, db_session) -> None:
        img1 = ImageFactory(session=db_session, sha256="hash1")
        img2 = ImageFactory(session=db_session, sha256="hash2")
        db_session.flush()
        assert img1.id != img2.id

    def test_dataset_not_unique(self, db_session) -> None:
        """Multiple images can share the same dataset."""
        img1 = ImageFactory(session=db_session, dataset="shared")
        img2 = ImageFactory(session=db_session, dataset="shared")
        db_session.flush()
        assert img1.dataset == img2.dataset


class TestImageFields:
    def test_nullable_fields(self, db_session) -> None:
        img = Image(sha256="minimal-hash")
        db_session.add(img)
        db_session.flush()

        db_session.refresh(img)
        assert img.dataset is None
        assert img.width is None
        assert img.s3_key is None
        assert img.phash is None
