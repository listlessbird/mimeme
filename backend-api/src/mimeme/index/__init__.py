from mimeme.index.client import Client
from mimeme.index.model import (
    Activated,
    ActivateInput,
    Build,
    BuildCall,
    BuildSpec,
    Built,
    BuiltFile,
    Embedding,
    Encoder,
    File,
    LocalEmbedding,
    Manifest,
    Phase,
    Prepared,
    PreparedBuild,
    PrepareInput,
    Result,
    Snapshot,
    State,
    Trigger,
    WorkflowInput,
    WorkflowResult,
)
from mimeme.index.state import InvalidTransition, transition


async def prepare(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import prepare as run

    return await run(*args, **kwargs)


async def build(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import build as run

    return await run(*args, **kwargs)


async def activate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import activate as run

    return await run(*args, **kwargs)


async def collect(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import collect as run

    return await run(*args, **kwargs)


async def validate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import validate as run

    return await run(*args, **kwargs)


async def cleanup_incomplete(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import cleanup_incomplete as run

    return await run(*args, **kwargs)


async def reconcile(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    from mimeme.index.ops import reconcile as run

    return await run(*args, **kwargs)


__all__ = [
    "ActivateInput",
    "Activated",
    "Build",
    "Client",
    "BuildCall",
    "BuildSpec",
    "Built",
    "BuiltFile",
    "Embedding",
    "Encoder",
    "File",
    "LocalEmbedding",
    "InvalidTransition",
    "Manifest",
    "Phase",
    "PrepareInput",
    "PreparedBuild",
    "Prepared",
    "Result",
    "Snapshot",
    "State",
    "Trigger",
    "WorkflowInput",
    "WorkflowResult",
    "activate",
    "build",
    "cleanup_incomplete",
    "collect",
    "prepare",
    "reconcile",
    "transition",
    "validate",
]
