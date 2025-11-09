from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from ..config import CFG
from ..database import session_scope
from ..repositories import images as images_repo
from ..storage import (
    build_object_key,
    ensure_bucket_exists,
    head_object,
    upload_file,
)

console = Console()


def upload_command(
    root: Optional[str] = typer.Argument(None, help="Directory with raw memes (default: data/raw_memes)"),
    dry_run: bool = typer.Option(False, help="Only print planned uploads"),
):
    """Upload local images to S3 object storage."""
    base = Path(root or CFG.image_root).resolve()
    ensure_bucket_exists()

    with session_scope() as session:
        rows = images_repo.get_all_basic(session)

        to_upload: list[tuple[int, str, Path, str]] = []
        for r in rows:
            img_id, sha256, rel_path, s3_key, s3_etag = r
            local = base / rel_path
            if not local.exists():
                continue
            key = build_object_key(sha256, rel_path)
            remote_etag = head_object(key)
            if remote_etag and remote_etag == s3_etag:
                continue
            to_upload.append((img_id, sha256, local, key))

        if dry_run:
            for _, sha, p, k in to_upload:
                console.print(f"would upload {p} -> {k}")
            return

        uploaded = 0

        with (
            session_scope() as session,
            Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "{task.completed}/{task.total}",
                TimeElapsedColumn(),
                transient=False,
                console=console,
            ) as progress,
        ):
            task_id = progress.add_task("Uploading images", total=len(to_upload))

            for img_id, sha, local, key in to_upload:
                etag = upload_file(local, key)
                images_repo.set_s3_fields(session, img_id, key, etag)
                uploaded += 1
                progress.update(task_id, advance=1)
        console.print(f"[bold green]Uploaded {uploaded} images[/bold green]")
