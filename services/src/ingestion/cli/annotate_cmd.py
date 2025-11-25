import typer
from rich.console import Console

from ..annotate import annotate_batch

console = Console()


def annotate_command(
    batch_size: int = typer.Option(64, help="How many imgs to annotate in a single batch"),
    model: str = typer.Option("moondream2", help="Vision model to use"),
):
    n = annotate_batch(batch_size=batch_size, model_name=model)
    console.print(f"[green]Annotated {n} images(s).[/green]")
