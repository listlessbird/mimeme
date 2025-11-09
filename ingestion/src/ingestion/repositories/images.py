from __future__ import annotations
from typing import Iterable, List, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..orm import Image


def bulk_upsert(session: Session, records: Iterable[dict]) -> None:
    recs = list(records)
    if not recs:
        return
    stmt = sqlite_insert(Image).values(recs)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Image.sha256],
        set_={
            "rel_path": stmt.excluded.rel_path,
            "width": func.coalesce(stmt.excluded.width, Image.width),
            "height": func.coalesce(stmt.excluded.height, Image.height),
            "format": func.coalesce(stmt.excluded.format, Image.format),
            "phash": func.coalesce(stmt.excluded.phash, Image.phash),
        },
    )
    session.execute(stmt)


def count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(Image)).scalar_one()


def list_recent(session: Session, limit: int) -> List[Image]:
    rows = session.execute(select(Image).order_by(Image.id.desc()).limit(limit)).scalars().all()
    return rows


def get_all_basic(session: Session) -> List[Tuple[int, str, str, str | None, str | None]]:
    stmt = select(Image.id, Image.sha256, Image.rel_path, Image.s3_key, Image.s3_etag)
    return session.execute(stmt).all()


def get_with_s3_key(session: Session) -> List[Tuple[int, str]]:
    stmt = select(Image.id, Image.s3_key).where(Image.s3_key.is_not(None)).order_by(Image.id)
    return session.execute(stmt).all()


def set_s3_fields(session: Session, image_id: int, key: str, etag: str) -> None:
    stmt = (
        update(Image)
        .where(Image.id == image_id)
        .values(s3_key=key, s3_etag=etag)
    )
    session.execute(stmt)
