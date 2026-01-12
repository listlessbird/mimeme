import hashlib
from pathlib import Path

import imagehash
from PIL import Image


def compute_sha256(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)

    return h.hexdigest()


def compute_phash(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            ph = imagehash.phash(img)
            return str(ph)
    except Exception:
        return None


def get_image_info(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as img:
            w, h = img.size
            fmt = (img.format or "").lower() if img.format else None
            return w, h, fmt
    except Exception:
        return None, None, None
