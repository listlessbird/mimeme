from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from .config import CFG
from .orm import Annotation, Artifact, Processing, session_scope
from .orm import Image as ORMImage
from .vision.base import create_vision_model
from .vision.moondream2 import MoonDream2Config


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _ensure_processing_row(sess, img: ORMImage):
    if not img.processing:
        p = Processing(image_id=img.id)
        sess.add(p)
        sess.flush()
        sess.refresh(img)


def _get_or_create_annotation(sess, img: ORMImage) -> Annotation:
    """Get existing annotation or create a new one."""
    ann = sess.query(Annotation).filter_by(image_id=img.id).first()
    if not ann:
        ann = Annotation(image_id=img.id)
        sess.add(ann)
        sess.flush()
    return ann


def _add_artifact_if_not_exists(sess, image_id: int, kind: str, model_version: str) -> None:
    """Add artifact only if it doesn't already exist."""
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
                path="",
                checksum="",
                created_at=now_iso(),
            )
        )


def annotate_batch(
    batch_size: int = 32,
    model_name: str = "moondream2",
    show_progress: bool = True,
):
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    model = create_vision_model(model_name, cfg=MoonDream2Config())
    console = Console()

    done = 0
    failed = 0
    recent_results = []

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

        if not images:
            if show_progress:
                console.print("[yellow]No images need annotation[/yellow]")
            return 0

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[green]✓ {task.fields[done]}"),
                TextColumn("[red]✗ {task.fields[failed]}"),
                TimeElapsedColumn(),
                console=console,
            )

            with progress:
                task = progress.add_task(
                    "Annotating images",
                    total=len(images),
                    done=0,
                    failed=0,
                )

                for img in images:
                    short_path = str(Path(img.rel_path).name)[:40]
                    progress.update(task, description=f"Processing {short_path}")

                    try:
                        result = _process_single_image(sess, model, img)
                        if result["success"]:
                            done += 1
                            progress.update(task, advance=1, done=done, failed=failed)

                            # Keep last 5 results for summary
                            if len(recent_results) >= 5:
                                recent_results.pop(0)
                            recent_results.append(result)
                        else:
                            failed += 1
                            progress.update(task, advance=1, done=done, failed=failed)
                    except Exception as e:
                        failed += 1
                        progress.update(task, advance=1, done=done, failed=failed)
                        console.print(f"[red]✗ Failed image {img.id}: {e}[/red]")
        else:
            for img in images:
                try:
                    result = _process_single_image(sess, model, img)
                    if result["success"]:
                        done += 1
                        if len(recent_results) >= 5:
                            recent_results.pop(0)
                        recent_results.append(result)
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    print(f"Annotate failed for image {img.id}: {e}")

        if show_progress and recent_results:
            console.print("\n[bold]Recent Annotations:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", width=5)
            table.add_column("Image", width=25)
            table.add_column("Caption", width=50)
            table.add_column("OCR", width=30)

            for result in recent_results:
                caption = (
                    result.get("caption", "")[:47] + "..."
                    if len(result.get("caption", "")) > 50
                    else result.get("caption", "")
                )
                ocr = (
                    result.get("ocr", "")[:27] + "..."
                    if len(result.get("ocr", "")) > 30
                    else result.get("ocr", "")
                )

                table.add_row(
                    str(result["id"]),
                    str(Path(result["path"]).name)[:25],
                    caption or "[dim]—[/dim]",
                    ocr or "[dim]—[/dim]",
                )

            console.print(table)

    return done


def _process_single_image(sess, model, img: ORMImage) -> dict:
    """Process a single image and return results."""
    try:
        _ensure_processing_row(sess, img)

        p = Path(CFG.image_root) / img.rel_path

        if not p.exists():
            img.processing.ocr_status = img.processing.ocr_status or "failed"
            img.processing.caption_status = img.processing.caption_status or "failed"
            sess.commit()
            return {"success": False, "reason": "file_not_found"}

        pil = Image.open(p).convert("RGB")

        needs_caption = img.processing.caption_status in (None, "pending")
        needs_ocr = img.processing.ocr_status in (None, "pending")

        if not needs_caption and not needs_ocr:
            return {"success": False, "reason": "already_processed"}

        caption_text = None
        ocr_text = None

        if needs_caption:
            cap = model.caption(pil, length="normal")
            ann = _get_or_create_annotation(sess, img)
            ann.caption_text = cap.caption
            caption_text = cap.caption
            img.processing.caption_status = "done"
            img.processing.caption_model = cap.model
            img.processing.caption_updated_at = now_iso()
            _add_artifact_if_not_exists(sess, img.id, "caption", cap.model)

        if needs_ocr:
            ocr = model.ocr(pil)
            ann = _get_or_create_annotation(sess, img)
            ann.ocr_text = ocr.text
            ocr_text = ocr.text
            img.processing.ocr_status = "done"
            img.processing.ocr_model = ocr.model
            img.processing.ocr_updated_at = now_iso()
            _add_artifact_if_not_exists(sess, img.id, "ocr", ocr.model)

        sess.commit()

        return {
            "success": True,
            "id": img.id,
            "path": img.rel_path,
            "caption": caption_text,
            "ocr": ocr_text,
        }

    except Exception as e:
        sess.rollback()
        try:
            # Mark as failed in a separate transaction
            if img.processing:
                if img.processing.caption_status in (None, "pending"):
                    img.processing.caption_status = "failed"
                if img.processing.ocr_status in (None, "pending"):
                    img.processing.ocr_status = "failed"
            sess.commit()
        except SQLAlchemyError:
            sess.rollback()
        raise e
