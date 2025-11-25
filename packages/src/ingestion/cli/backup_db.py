import gzip
import shutil
import sqlite3
import time
from pathlib import Path

import typer
from rich.console import Console

from ..config import CFG
from ..database import get_engine
from ..storage import ensure_bucket_exists, upload_file

console = Console()


def backup_db_command(
    compress: bool = typer.Option(True, help="Gzip the backup before upload"),
):
    """Create a snapshot of the database and upload to S3."""
    db_path = Path(CFG.db_path).resolve()
    snap = db_path.with_name(f"{db_path.stem}.snapshot-{int(time.time())}.sqlite3")

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
