from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from ..config import CFG
from ..database import session_scope
from ..repositories import images as images_repo
from ..pipeline import walk_images, process_paths_parallel, batched

console = Console()


def scan_command(
    root: Optional[str] = typer.Argument(
        None, help="Directory with raw memes (default: data/raw_memes)"
    ),
    workers: int = typer.Option(CFG.workers, help="Number of workers to use"),
    batch_size: int = typer.Option(CFG.batch_size, help="Number of images to process in a batch"),
    estimate: bool = typer.Option(True, help="Estimate the number of images to process"),
):
    base = Path(root or CFG.image_root).resolve()
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
