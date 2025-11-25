from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image
import imagehash


def compute_sha256(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as img:
            ph = imagehash.phash(img)
            return str(ph)
    except Exception as e:
        print(f"Error computing phash for {path}: {e}")
        return None
