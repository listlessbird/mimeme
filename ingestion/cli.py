import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn


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

if __name__ == "__main__":
    app()