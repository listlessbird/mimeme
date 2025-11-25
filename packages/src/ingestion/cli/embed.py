import typer
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from ..embeddings.base import EmbedderConfig
from ..embeddings.pipeline import run_embedding_loop

console = Console()


def embed_command(
    batch_size: int = typer.Option(8, help="Batch size for embedding"),
    model: str = typer.Option(
        "google/siglip2-base-patch16-naflex", help="HuggingFace vision model name"
    ),
    limit: int = typer.Option(None, help="Limit number of images to process"),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)"
    ),
):
    """Run the embedding pipeline for pending images."""
    cfg = EmbedderConfig(
        image_model=model,
        device=device,
    )
    cfg.batch_size = batch_size

    console.print(
        f"[green]Starting embedding pipeline with model={model}, device={device}, batch_size={batch_size}[/green]"
    )

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        transient=False,
        console=console,
    ) as progress:
        task_id = None
        
        for n, total in run_embedding_loop(cfg, limit=limit):
            if task_id is None:
                task_id = progress.add_task("Embedding", total=total)
            else:
                progress.update(task_id, advance=n)

    console.print("[green]Embedding pipeline finished.[/green]")