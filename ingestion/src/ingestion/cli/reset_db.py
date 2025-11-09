import typer
from rich.console import Console

from ..database import session_scope
from ..orm import Image, Processing, Annotation, Artifact, IndexBuild

console = Console()


def reset_db_command(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset the database by removing all rows from all tables."""
    if not confirm:
        response = typer.confirm(
            "This will DELETE ALL DATA from the database. Are you sure?",
            default=False,
        )
        if not response:
            console.print("[yellow]Operation cancelled[/yellow]")
            raise typer.Exit(0)

    with session_scope() as session:
        # Delete in order to respect foreign keys
        deleted_artifacts = session.query(Artifact).delete()
        deleted_annotations = session.query(Annotation).delete()
        deleted_processing = session.query(Processing).delete()
        deleted_index_builds = session.query(IndexBuild).delete()
        deleted_images = session.query(Image).delete()

        session.commit()

        console.print(f"[green]Database reset complete:[/green]")
        console.print(f"  - Deleted {deleted_images} images")
        console.print(f"  - Deleted {deleted_processing} processing records")
        console.print(f"  - Deleted {deleted_annotations} annotations")
        console.print(f"  - Deleted {deleted_artifacts} artifacts")
        console.print(f"  - Deleted {deleted_index_builds} index builds")
