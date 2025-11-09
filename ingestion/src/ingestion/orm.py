from __future__ import annotations
from sqlalchemy import Column, Integer, Text, String, ForeignKey, text, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
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


Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    sha256 = Column(Text, unique=True, nullable=False)
    rel_path = Column(Text, nullable=False)
    s3_key = Column(Text)
    s3_etag = Column(Text)
    width = Column(Integer)
    height = Column(Integer)
    format = Column(Text)
    phash = Column(Text)
    created_at = Column(Text, server_default=text("CURRENT_TIMESTAMP"))

    processing = relationship("Processing", uselist=False, back_populates="image")
    annotations = relationship("Annotation", uselist=False, back_populates="image")
    artifacts = relationship("Artifact", back_populates="image")


class Processing(Base):
    __tablename__ = "processing"
    image_id = Column(Integer, ForeignKey("images.id"), primary_key=True)
    ocr_status = Column(String, default="pending")  # pending|running|done|failed
    ocr_model = Column(Text)
    ocr_updated_at = Column(Text)
    caption_status = Column(String, default="pending")
    caption_model = Column(Text)
    caption_updated_at = Column(Text)
    embed_status = Column(String, default="pending")
    embed_model = Column(Text)
    embed_dim = Column(Integer)
    embed_updated_at = Column(Text)

    image = relationship("Image", back_populates="processing")


class Annotation(Base):
    __tablename__ = "annotations"
    image_id = Column(Integer, ForeignKey("images.id"), primary_key=True)
    ocr_text = Column(Text)
    caption_text = Column(Text)
    tags = Column(Text)
    image = relationship("Image", back_populates="annotations")


class Artifact(Base):
    __tablename__ = "artifacts"
    image_id = Column(Integer, ForeignKey("images.id"), primary_key=True)
    kind = Column(Text, primary_key=True)  # 'ocr'|'caption'|'embed'|'thumb'
    model_version = Column(Text, primary_key=True)
    path = Column(Text)
    checksum = Column(Text)
    created_at = Column(Text, server_default=text("CURRENT_TIMESTAMP"))

    image = relationship("Image", back_populates="artifacts")
    __table_args__ = (UniqueConstraint("image_id", "kind", "model_version"),)


class IndexBuild(Base):
    __tablename__ = "index_builds"
    id = Column(Integer, primary_key=True)
    faiss_path = Column(Text)
    faiss_trained_on = Column(Text)
    clip_model = Column(Text)
    num_vectors = Column(Integer)
    created_at = Column(Text, server_default=text("CURRENT_TIMESTAMP"))
