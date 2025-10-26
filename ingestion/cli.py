import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from storage import build_object_key, ensure_bucket_exists, head_object, upload_file


from .config import CFG
from .db import connect, bulk_upsert_images
from .pipeline import walk_images, process_paths_parallel, batched

app = typer.Typer()
console = Console()

app.command()

@app.command()
def scan(
    root: str = typer.Argument(...,help="Directory with raw memes"),
    workers: int = typer.Option(CFG.workers,help="Number of workers to use"),
    batch_size: int = typer.Option(CFG.batch_size,help="Number of images to process in a batch"),
    estimate: bool = typer.Option(True,help="Estimate the number of images to process")
):
    base = Path(root).resolve()
    if not base.exists():
        console.print(f"[red]Directory {base} does not exist[/red]")
        raise typer.Exit(2)

    paths_iter = list(walk_images(base)) if estimate else walk_images(base)

    total = len(paths_iter) if estimate else None

    console.rule(f"[bold green]Scanning {base}[/bold green]")

    rec_iter = process_paths_parallel(paths_iter, base, workers) if estimate else process_paths_parallel(paths_iter, base, workers)

    with connect() as conn, Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        transient=False,
        console=console
    ) as progress:
        task_id = progress.add_task("Processing images", total=total)

        for chunk in batched(rec_iter, batch_size):
            bulk_upsert_images(conn, [rec.as_dict() for rec in chunk])
            progress.update(task_id, advance=len(chunk))
    
    console.print(f"[bold green]Scanned {total} images[/bold green]")

@app.command()
def verify():
    from rich.table import Table

    with connect() as con:
        row = con.execute("SELECT COUNT(*) AS c FROM images").fetchone()
        cnt = row["c"]
        console.print(f"[bold]images[/bold]: {cnt}")

        table = Table(show_lines=True)
        for col in ("id", "sha256", "rel_path", "width", "height", "format", "phash"):
            table.add_column(col)

        for r in con.execute("SELECT id, sha256, rel_path, width, height, format, phash FROM images ORDER BY id DESC LIMIT 5"):
            table.add_row(*(str(r[c]) if r[c] is not None else "" for c in ("id", "sha256", "rel_path", "width", "height", "format", "phash")))
        console.print(table)

@app.command()
def upload(
    root: str = typer.Argument(..., help="Directory with raw memes"),
    dry_run: bool = typer.Option(False, help="Only print planned uploads")
):
    base = Path(root).resolve()
    ensure_bucket_exists()

    with connect() as con:
        rows = con.execute("SELECT id, sha256, rel_path, s3_key, s3_etag FROM images").fetchall()

        to_upload: list[tuple[int, str, Path, str]] = []
        for r in rows:
            local = Path(root) / r["rel_path"]
            if not local.exists():
                continue
            key = build_object_key(r["sha256"], r["rel_path"])
            remote_etag = head_object(key)
            if remote_etag and remote_etag == r["s3_etag"]:
                continue
            to_upload.append((r["id"], r["sha256"], local, key))

        if dry_run:
            for _, sha, p, k in to_upload:
                console.print(f"would upload {p} -> {k}")
            return

        uploaded = 0
        
        with connect() as conn, Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            transient=False,
            console=console
        ) as progress:
            task_id = progress.add_task("Uploading images", total=len(to_upload))

            for img_id, sha, local, key in to_upload:
                etag = upload_file(local, key)
                conn.execute(
                    "UPDATE images SET s3_key = :key, s3_etag = :etag WHERE id = :img_id",
                    {"key": key, "etag": etag, "img_id": img_id}
                )
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
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(snap)
    with dst:
        src.backup(dst)
    src.close(); dst.close()

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

if __name__ == "__main__":
    app()