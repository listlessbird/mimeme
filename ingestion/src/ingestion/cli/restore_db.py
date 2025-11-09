import gzip
import shutil
from pathlib import Path

import typer
from rich.console import Console

from ..config import CFG
from ..storage import list_objects, download_file

console = Console()


def restore_db_command(
    dest: str = typer.Option(None, help="Destination directory for restored DB"),
):
    """Download and restore the latest database backup from S3."""
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
