from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from api.config import settings


def get_engine(echo: bool = False):
    connect_args = {}
    if settings.db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        settings.db_url,
        echo=echo,
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


SessionLocal = sessionmaker(
    bind=get_engine(),
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    sess = SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    sha256: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True, default=None)
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    s3_etag: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    format: Mapped[str | None] = mapped_column(String(10), nullable=True, default=None)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    phash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, init=False
    )

    processing: Mapped[Processing | None] = relationship(back_populates="image", uselist=False, init=False, default=None)
    annotations: Mapped[Annotation | None] = relationship(back_populates="image", uselist=False, init=False, default=None)
    artifacts: Mapped[list[Artifact]] = relationship(back_populates="image", init=False, default_factory=list)


class Processing(Base):
    __tablename__ = "processing"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    ocr_status: Mapped[str | None] = mapped_column(String(20), default="pending")
    ocr_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    ocr_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    caption_status: Mapped[str | None] = mapped_column(String(20), default="pending")
    caption_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    caption_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    embed_status: Mapped[str | None] = mapped_column(String(20), default="pending")
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    embed_dim: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    embed_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    embed_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    image: Mapped[Image] = relationship(back_populates="processing", init=False, default=None)


class Annotation(Base):
    __tablename__ = "annotations"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    caption_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    image: Mapped[Image] = relationship(back_populates="annotations", init=False, default=None)


class Artifact(Base):
    __tablename__ = "artifacts"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    kind: Mapped[str] = mapped_column(String(20), primary_key=True)
    model_version: Mapped[str] = mapped_column(Text, primary_key=True)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, init=False
    )

    image: Mapped[Image] = relationship(back_populates="artifacts", init=False, default=None)

    __table_args__ = (
        UniqueConstraint(
            "image_id", "kind", "model_version", name="uq_artifacts_image_kind_version"
        ),
    )


class IndexBuild(Base):
    __tablename__ = "index_builds"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    index_type: Mapped[str | None] = mapped_column(String(20), nullable=True, default=None)
    num_vectors: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, init=False
    )
