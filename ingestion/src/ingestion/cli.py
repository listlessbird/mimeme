from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from .storage import (
    build_object_key,
    download_file,
    ensure_bucket_exists,
    head_object,
    list_objects,
    upload_file,
)


from .config import CFG
from .database import session_scope, get_engine
from .repositories import images as images_repo
from .pipeline import walk_images, process_paths_parallel, batched

app = typer.Typer()
console = Console()

app.command()


@app.command()
def scan(
    root: str = typer.Argument(..., help="Directory with raw memes"),
    workers: int = typer.Option(CFG.workers, help="Number of workers to use"),
    batch_size: int = typer.Option(
        CFG.batch_size, help="Number of images to process in a batch"
    ),
    estimate: bool = typer.Option(
        True, help="Estimate the number of images to process"
    ),
):
    base = Path(root).resolve()
    if not base.exists():
        console.print(f"[red]Directory {base} does not exist[/red]")
        raise typer.Exit(2)

    paths_iter = list(walk_images(base)) if estimate else walk_images(base)

    total = len(paths_iter) if estimate else None

    console.rule(f"[bold green]Scanning {base}[/bold green]")

    rec_iter = (
        process_paths_parallel(paths_iter, base, workers)
        if estimate
        else process_paths_parallel(paths_iter, base, workers)
    )

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
        task_id = progress.add_task("Processing images", total=total)

        for chunk in batched(rec_iter, batch_size):
            images_repo.bulk_upsert(session, [rec.as_dict() for rec in chunk])
            progress.update(task_id, advance=len(chunk))

    console.print(f"[bold green]Scanned {total} images[/bold green]")


@app.command()
def verify():
    from rich.table import Table

    with session_scope() as session:
        cnt = images_repo.count(session)
        console.print(f"[bold]images[/bold]: {cnt}")

        table = Table(show_lines=True)
        for col in ("id", "sha256", "rel_path", "width", "height", "format", "phash"):
            table.add_column(col)

        rows = images_repo.list_recent(session, 5)
        for r in rows:
            table.add_row(
                *(
                    str(getattr(r, c)) if getattr(r, c) is not None else ""
                    for c in (
                        "id",
                        "sha256",
                        "rel_path",
                        "width",
                        "height",
                        "format",
                        "phash",
                    )
                )
            )
        console.print(table)


@app.command()
def upload(
    root: str = typer.Argument(..., help="Directory with raw memes"),
    dry_run: bool = typer.Option(False, help="Only print planned uploads"),
):
    base = Path(root).resolve()
    ensure_bucket_exists()

    with session_scope() as session:
        rows = images_repo.get_all_basic(session)

        to_upload: list[tuple[int, str, Path, str]] = []
        for r in rows:
            img_id, sha256, rel_path, s3_key, s3_etag = r
            local = Path(root) / rel_path
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


@app.command()
def backup_db(
    compress: bool = typer.Option(True, help="Gzip the backup before upload"),
):
    import gzip, shutil, time

    db_path = Path(CFG.db_path).resolve()
    snap = db_path.with_name(f"{db_path.stem}.snapshot-{int(time.time())}.sqlite3")

    import sqlite3

    engine = get_engine()
    src = engine.raw_connection()
    dst = sqlite3.connect(snap)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()

    target = snap
    if compress:
        gz = Path(str(snap) + ".gz")
        with open(snap, "rb") as f_in, gzip.open(gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        target = gz

    ensure_bucket_exists()
    key = f"backups/{target.name}"
    etag = upload_file(target, key)

    console.print(f"[green]DB backup uploaded:[/green] {key} (etag {etag})")


@app.command()
def restore_db(
    dest: str = typer.Option(None, help="Destination directory for restored DB"),
):
    import gzip, shutil

    target_db = Path(dest) if dest else Path(CFG.db_path)
    objs = list_objects("backups/")
    if not objs:
        console.print(f"[red]No backup objects found in S3 under backups/[/red]")
        raise typer.Exit(2)

    latest_key = sorted((k for k, _ in objs))[-1]
    tmp = target_db.with_suffix(target_db.suffix + ".download")
    download_file(latest_key, tmp)

    if latest_key.endswith(".gz"):
        with gzip.open(tmp, "rb") as fi, open(target_db, "wb") as fo:
            shutil.copyfileobj(fi, fo)
            tmp.unlink(missing_ok=True)
    else:
        tmp.rename(target_db)
    console.print(f"[green]DB restored to:[/green] {target_db}")


@app.command()
def rehydrate(
    dest_root: str = typer.Argument(..., help="Directory to store rehydrated images"),
):
    base = Path(dest_root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    with session_scope() as session:
        rows = images_repo.get_with_s3_key(session)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=len(rows))

        for r in rows:
            _, key = r
            local = base.joinpath(*key.split("/"))
            download_file(key, local)
            progress.update(task, advance=1)
    console.print(f"[green]Rehydrated {len(rows)} images to {base}[/green]")


@app.command()
def annotate(
    batch_size: int = typer.Option(
        64, help="How many imgs to annotate in a single batch"
    ),
    model: str = typer.Option("moondream2", help="Vision model to use"),
):
    from .annotate import annotate_batch
    from rich.console import Console

    console = Console()
    n = annotate_batch(batch_size=batch_size, model_name=model)
    console.print(f"[green]Annotated {n} images(s).[/green]")


if __name__ == "__main__":
    app()

