

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ingestion.config import CFG
from ingestion.orm import Artifact, Image as ORMImage, session_scope, Annotation as ORMAnnotation, Processing

import numpy as np

ARTIFACT_ROOT = Path(CFG.db_path).parent / "artifacts"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_artifact_dir(model_name: str) -> Path:

    p = ARTIFACT_ROOT / model_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_artifact_if_not_exists(
    sess,
    image_id: int,
    kind: str,
    model_version: str,
    path: str
):

    existing = (
        sess.query(Artifact)
        .filter_by(image_id=image_id, kind=kind, model_version=model_version)
        .first()
    )
    if not existing:
        sess.add(
            Artifact(
                image_id=image_id,
                kind=kind,
                model_version=model_version,
                path=str(path),
                checksum="",
                created_at=now_iso()
            )
        )


def get_image_text(image_id: int) -> str:
    text_val = ""
    try:
        with session_scope() as sess:
            ann_row = sess.query(ORMAnnotation).filter_by(
                image_id=image_id).first()

            if ann_row:
                parts = []
                if ann_row.caption_text is not None:
                    parts.append(ann_row.caption_text)
                if ann_row.ocr_text is not None:
                    parts.append(ann_row.ocr_text)
                text_val = " ".join(parts)
    except Exception:
        pass
    return text_val


def get_pending_embeddings(
    limit: Optional[int] = None
) -> List[Tuple[int, str]]:
    with session_scope() as sess:
        q = (
            sess.query(ORMImage)
            .join(Processing, ORMImage.id == Processing.image_id, isouter=True)
            .filter(
                (Processing.embed_status == "pending") |
                (Processing.embed_status.is_(None))
            )
        )

        if limit:
            q = q.limit(limit)

        return [(r.id, r.rel_path) for r in q.all()]  # type:ignore


def mark_embedding_status(image_ids: List[int], status: str):
    with session_scope() as sess:
        for img_id in image_ids:
            proc = sess.query(Processing).filter_by(image_id=img_id).first()
            if not proc:
                proc = Processing(image_id=img_id)
                sess.add(proc)
                sess.flush()

            proc.embed_status = status  # type: ignore
        sess.commit()


def save_embeddings(
    image_id: int,
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    model_name: str,
    artifact_dir: Path
):

    with session_scope() as sess:
        img = sess.query(ORMImage).filter_by(id=image_id).first()
        sha = img.sha256 if img else str(image_id)

    img_path = artifact_dir / f"{sha}.npy"
    txt_path = artifact_dir / f"{sha}_text.npy"
    np.save(img_path, image_embedding)
    np.save(txt_path, text_embedding)

    with session_scope() as sess:
        rel_img_path = img_path.relative_to(
            Path(CFG.db_path).parent).as_posix()
        add_artifact_if_not_exists(
            sess, image_id, "embed", model_name, rel_img_path)

        proc = sess.query(Processing).filter_by(image_id=image_id).first()
        if not proc:
            proc = Processing(image_id=image_id)
            sess.add(proc)
            sess.flush()

        proc.embed_status = "done"  # type: ignore
        proc.embed_model = model_name  # type: ignore
        proc.embed_dim = int(image_embedding.shape[-1])  # type: ignore
        proc.embed_updated_at = now_iso()  # type: ignore
        sess.commit()
