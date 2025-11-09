import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func

from ..database import session_scope
from ..repositories import images as images_repo
from ..orm import Processing, Annotation

console = Console()


def verify_command():
    """Verify database contents and show processing status."""
    with session_scope() as session:
        cnt = images_repo.count(session)
        console.print(f"[bold]images[/bold]: {cnt}")

        # Check processing status summary
        status_query = (
            session.query(
                Processing.ocr_status,
                Processing.caption_status,
                func.count().label("count")
            )
            .group_by(Processing.ocr_status, Processing.caption_status)
        )

        console.print("\n[bold]Processing Status:[/bold]")
        for ocr, caption, count in status_query:
            ocr_str = ocr or "null"
            caption_str = caption or "null"
            status_color = "green" if ocr == "done" and caption == "done" else "yellow" if "pending" in [ocr, caption] else "red"
            console.print(f"  [{status_color}]ocr={ocr_str}, caption={caption_str}: {count}[/{status_color}]")

        # Check for stuck running states
        stuck_count = session.query(Processing).filter(
            (Processing.ocr_status == "running") | (Processing.caption_status == "running")
        ).count()
        if stuck_count > 0:
            console.print(f"  [red]⚠ {stuck_count} images stuck in 'running' state[/red]")

        # Check annotations
        ann_count = session.query(Annotation).count()
        console.print(f"\n[bold]annotations[/bold]: {ann_count}")

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
