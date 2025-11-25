from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from ..database import session_scope
from ..repositories import images as images_repo
from ..storage import download_file

console = Console()


def rehydrate_command(
    dest_root: str = typer.Argument(..., help="Directory to store rehydrated images"),
):
    """Download all images from S3 to local directory."""
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
