from __future__ import annotations
from typing import Optional, List
from sqlalchemy import Integer, Text, String, ForeignKey, text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy import create_engine
from contextlib import contextmanager
import os

from .config import CFG


def _sqlite_url() -> str:
    db_abs = os.path.abspath(CFG.db_path)
    return f"sqlite:///{db_abs}"


def get_engine(echo: bool = False):
    return create_engine(
        _sqlite_url(),
        echo=echo,
        future=True,
        connect_args={"check_same_thread": False},
    )


SessionLocal = sessionmaker(
    bind=get_engine(),
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,
)


@contextmanager
def session_scope():
    sess = SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(Text, unique=True)
    rel_path: Mapped[str] = mapped_column(Text)
    s3_key: Mapped[Optional[str]] = mapped_column(Text)
    s3_etag: Mapped[Optional[str]] = mapped_column(Text)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    format: Mapped[Optional[str]] = mapped_column(Text)
    phash: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[str]] = mapped_column(
        Text, server_default=text("CURRENT_TIMESTAMP")
    )

    processing: Mapped[Optional["Processing"]] = relationship(back_populates="image")
    annotations: Mapped[Optional["Annotation"]] = relationship(back_populates="image")
    artifacts: Mapped[List["Artifact"]] = relationship(back_populates="image")


class Processing(Base):
    __tablename__ = "processing"
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), primary_key=True)
    ocr_status: Mapped[Optional[str]] = mapped_column(String, default="pending")
    ocr_model: Mapped[Optional[str]] = mapped_column(Text)
    ocr_updated_at: Mapped[Optional[str]] = mapped_column(Text)
    caption_status: Mapped[Optional[str]] = mapped_column(String, default="pending")
    caption_model: Mapped[Optional[str]] = mapped_column(Text)
    caption_updated_at: Mapped[Optional[str]] = mapped_column(Text)
    embed_status: Mapped[Optional[str]] = mapped_column(String, default="pending")
    embed_model: Mapped[Optional[str]] = mapped_column(Text)
    embed_dim: Mapped[Optional[int]] = mapped_column(Integer)
    embed_updated_at: Mapped[Optional[str]] = mapped_column(Text)

    image: Mapped["Image"] = relationship(back_populates="processing")


class Annotation(Base):
    __tablename__ = "annotations"
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), primary_key=True)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text)
    caption_text: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[str]] = mapped_column(Text)

    image: Mapped["Image"] = relationship(back_populates="annotations")


class Artifact(Base):
    __tablename__ = "artifacts"
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), primary_key=True)
    kind: Mapped[str] = mapped_column(Text, primary_key=True)  # 'ocr'|'caption'|'embed'|'thumb'
    model_version: Mapped[str] = mapped_column(Text, primary_key=True)
    path: Mapped[Optional[str]] = mapped_column(Text)
    checksum: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[str]] = mapped_column(
        Text, server_default=text("CURRENT_TIMESTAMP")
    )

    image: Mapped["Image"] = relationship(back_populates="artifacts")
    __table_args__ = (UniqueConstraint("image_id", "kind", "model_version"),)


class IndexBuild(Base):
    __tablename__ = "index_builds"
    id: Mapped[int] = mapped_column(primary_key=True)
    faiss_path: Mapped[Optional[str]] = mapped_column(Text)
    faiss_trained_on: Mapped[Optional[str]] = mapped_column(Text)
    clip_model: Mapped[Optional[str]] = mapped_column(Text)
    num_vectors: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[str]] = mapped_column(
        Text, server_default=text("CURRENT_TIMESTAMP")
    )
