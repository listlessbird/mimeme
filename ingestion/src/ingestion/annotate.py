from __future__ import annotations
from datetime import datetime
from pathlib import Path
from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from .orm import Annotation, session_scope, Image as ORMImage, Processing, Artifact
from .vision.base import create_vision_model
from .vision.moondream2 import MoonDream2Config
from .config import CFG


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _ensure_processing_row(sess, img: ORMImage):
    if not img.processing:
        sess.add(Processing(image_id=img.id))
        sess.flush()


def annotate_batch(
    batch_size: int = 32,
    model_name: str = "moondream2",
):
    model = create_vision_model(model_name, cfg=MoonDream2Config())

    done = 0
    with session_scope() as sess:
        q = (
            select(ORMImage)
            .join(Processing, ORMImage.id == Processing.image_id, isouter=True)
            .where(
                (Processing.ocr_status == "pending")
                | (Processing.caption_status == "pending")
                | (Processing.ocr_status.is_(None))
                | (Processing.caption_status.is_(None))
            )
            .limit(batch_size)
        )
        images = list(sess.execute(q).scalars())

        for img in images:
            try:
                _ensure_processing_row(sess, img)

                p = Path(CFG.db_path).parent.parent / img.rel_path

                if not p.exists():
                    img.processing.ocr_status = img.processing.ocr_status or "failed"
                    img.processing.caption_status = (
                        img.processing.caption_status or "failed"
                    )
                    continue

                pil = Image.open(p).convert("RGB")

                if img.processing.caption_status in (None, "pending"):
                    img.processing.caption_status = "running"
                    sess.commit()
                    cap = model.caption(pil, length="normal")

                    ann = img.annotations or Annotation(image_id=img.id)
                    ann.caption_text = cap.caption
                    sess.add(ann)
                    img.processing.caption_status = "done"
                    img.processing.caption_model = cap.model
                    img.processing.caption_updated_at = now_iso()

                    sess.add(
                        Artifact(
                            image_id=img.id,
                            kind="caption",
                            model_version=cap.model,
                            path="",
                            checksum="",
                            created_at=now_iso(),
                        )
                    )
                    sess.commit()

                if img.processing.ocr_status in (None, "pending"):
                    img.processing.ocr_status = "running"
                    sess.commit()
                    ocr = model.ocr(pil)
                    ann = img.annotations or Annotation(image_id=img.id)
                    ann.ocr_text = ocr.text
                    sess.add(ann)

                    img.processing.ocr_status = "done"
                    img.processing.ocr_model = ocr.model
                    img.processing.ocr_updated_at = now_iso()

                    sess.add(
                        Artifact(
                            image_id=img.id,
                            kind="ocr",
                            model_version=ocr.model,
                            path="",
                            checksum="",
                            created_at=now_iso(),
                        )
                    )
                    sess.commit()
                done += 1

            except Exception as e:
                try:
                    if img.processing:
                        if img.processing.caption_status == "running":
                            img.processing.caption_status = "failed"
                        if img.processing.ocr_status == "running":
                            img.processing.ocr_status = "failed"

                    sess.commit()
                except SQLAlchemyError:
                    sess.rollback()
                print(f"Annotate failed for image {img.id}: {e}")
    return done
