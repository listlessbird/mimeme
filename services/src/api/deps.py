from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import settings
from api.services.indexer import FaissIndexManager
from api.services.search import SearchService
from api.services.storage import StorageService, get_storage_service


@lru_cache
def get_index_manager() -> FaissIndexManager:
    return FaissIndexManager(index_dir=settings.index_dir, db_url=settings.db_url)


IndexManagerDep = Annotated[FaissIndexManager, Depends(get_index_manager)]


@lru_cache
def get_engine():
    connect_args = {}
    if settings.db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        settings.db_url,
        connect_args=connect_args,
        echo=settings.debug,
        future=True,
    )
    return engine


@lru_cache
def get_session_factory():
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        future=True,
        expire_on_commit=False,
    )


def get_db() -> Iterator[Session]:
    SessionLocal = get_session_factory()
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        index_manager=get_index_manager(),
        model_name=settings.embed_model,
        device=settings.embed_device,
    )


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_storage() -> StorageService:
    return get_storage_service()


StorageDep = Annotated[StorageService, Depends(get_storage)]
