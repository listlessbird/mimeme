"""Factory Boy factories for all ORM models.

Usage in tests::

    def test_something(db_session):
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any, cast

import factory
from sqlalchemy.orm import Session

from shared.models.orm import (
    Annotation,
    Artifact,
    Image,
    IndexBuild,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    Processing,
    ProcessingStatus,
)


class _BaseMeta:
    sqlalchemy_session = None
    sqlalchemy_session_persistence = "flush"


class JobFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = Job

    id = factory.LazyFunction(lambda: f"test-{uuid.uuid4().hex[:12]}")
    type = JobType.INGEST
    status = JobStatus.PENDING
    progress = 0.0

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> Job:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class IngestURLFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = IngestURL

    url = factory.Sequence(lambda n: f"https://example.com/image_{n}.jpg")
    status = ProcessingStatus.PENDING
    job = factory.SubFactory(JobFactory)
    job_id = factory.LazyAttribute(lambda o: o.job.id)

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> IngestURL:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class ImageFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = Image

    sha256 = factory.LazyFunction(lambda: hashlib.sha256(os.urandom(32)).hexdigest())
    dataset = "test-dataset"
    original_filename = factory.Sequence(lambda n: f"image_{n}.jpg")
    s3_key = factory.LazyAttribute(lambda o: f"images/test/{o.sha256[:12]}.jpg")
    width = 800
    height = 600
    format = "jpeg"
    file_size = 102400
    phash = factory.LazyFunction(lambda: hashlib.md5(os.urandom(16)).hexdigest()[:16])

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> Image:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class ProcessingFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = Processing

    image = factory.SubFactory(ImageFactory)
    image_id = factory.LazyAttribute(lambda o: o.image.id)
    ocr_status = ProcessingStatus.PENDING
    caption_status = ProcessingStatus.PENDING
    embed_status = ProcessingStatus.PENDING

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> Processing:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class AnnotationFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = Annotation

    image = factory.SubFactory(ImageFactory)
    image_id = factory.LazyAttribute(lambda o: o.image.id)
    caption_text = factory.Sequence(lambda n: f"A funny meme about cats #{n}")
    ocr_text = factory.Sequence(lambda n: f"Sample OCR text #{n}")

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> Annotation:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class ArtifactFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = Artifact

    image = factory.SubFactory(ImageFactory)
    image_id = factory.LazyAttribute(lambda o: o.image.id)
    kind = "embedding"
    model_version = "siglip2-base"
    s3_key = factory.Sequence(lambda n: f"artifacts/test/artifact_{n}.bin")

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> Artifact:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


class IndexBuildFactory(factory.alchemy.SQLAlchemyModelFactory):  # ty: ignore[possibly-missing-attribute]
    class Meta(_BaseMeta):
        model = IndexBuild

    version = factory.LazyFunction(lambda: f"v-{uuid.uuid4().hex[:8]}")
    s3_key = factory.LazyAttribute(lambda o: f"indexes/{o.version}/index.faiss")
    embed_model = "google/siglip2-base-patch16-naflex"
    index_type = "flat"
    num_vectors = 1000
    dimension = 768
    is_active = False

    @classmethod
    def _create(
        cls, model_class: Any, *args: object, session: Session | None = None, **kwargs: object
    ) -> IndexBuild:
        if session is not None:
            cls._meta.sqlalchemy_session = session  # ty: ignore[invalid-assignment]
        return super()._create(model_class, *args, **kwargs)


def create_job(*, session: Session, **kwargs: object) -> Job:
    return cast(Job, JobFactory(session=session, **kwargs))


def create_ingest_url(*, session: Session, **kwargs: object) -> IngestURL:
    return cast(IngestURL, IngestURLFactory(session=session, **kwargs))


def create_image(*, session: Session, **kwargs: object) -> Image:
    return cast(Image, ImageFactory(session=session, **kwargs))


def create_processing(*, session: Session, **kwargs: object) -> Processing:
    return cast(Processing, ProcessingFactory(session=session, **kwargs))


def create_annotation(*, session: Session, **kwargs: object) -> Annotation:
    return cast(Annotation, AnnotationFactory(session=session, **kwargs))


def create_artifact(*, session: Session, **kwargs: object) -> Artifact:
    return cast(Artifact, ArtifactFactory(session=session, **kwargs))


def create_index_build(*, session: Session, **kwargs: object) -> IndexBuild:
    return cast(IndexBuild, IndexBuildFactory(session=session, **kwargs))
