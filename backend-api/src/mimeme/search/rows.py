from __future__ import annotations

from sqlalchemy import select

from mimeme.db import Db
from mimeme.db.schema import Annotation
from mimeme.db.schema import Image as ImageRow
from mimeme.search.run import Projection


class SqlRows:
    def __init__(self, db: Db) -> None:
        self._db = db

    async def fetch(self, image_ids: list[int]) -> dict[int, Projection]:
        if not image_ids:
            return {}
        async with self._db.read_session() as session:
            result = await session.execute(
                select(ImageRow, Annotation)
                .outerjoin(Annotation, Annotation.image_id == ImageRow.id)
                .where(ImageRow.id.in_(image_ids))
            )
        return {
            image.id: Projection(
                id=image.id,
                sha256=image.sha256,
                media_key=image.s3_key,
                caption=annotation.caption_text if annotation else None,
                ocr_text=annotation.ocr_text if annotation else None,
                width=image.width,
                height=image.height,
            )
            for image, annotation in result.all()
        }
