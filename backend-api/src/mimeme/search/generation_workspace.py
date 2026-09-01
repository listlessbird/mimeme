from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

_MARKER = ".mimeme-search-generation"


class Workspace:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._released = False


def prepare(parent: Path, version: str) -> Path:
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"search-{version}-", dir=parent))
    (root / _MARKER).write_text("owned by mimeme search\n", encoding="utf-8")
    return root


def claim(root: Path, artifact_paths: Iterable[Path]) -> Workspace:
    resolved = _validate(root)
    for path in artifact_paths:
        artifact = path.resolve(strict=True)
        if artifact.parent != resolved:
            raise ValueError("search artifacts must be direct children of their workspace")
    return Workspace(resolved)


def release(workspace: Workspace) -> None:
    if workspace._released:
        return
    discard(workspace._root)
    workspace._released = True


def discard(root: Path) -> None:
    try:
        resolved = _validate(root)
    except FileNotFoundError:
        return
    shutil.rmtree(resolved)


def _validate(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("search workspace must be an absolute non-symlink path")
    resolved = root.resolve(strict=True)
    if resolved == Path(resolved.anchor) or not resolved.is_dir():
        raise ValueError("search workspace is not a safe directory")
    marker = resolved / _MARKER
    if not marker.is_file():
        raise ValueError("search workspace ownership marker is missing")
    return resolved
