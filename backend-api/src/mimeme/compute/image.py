from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
from PIL import Image

from mimeme.compute.model import ImageInfo, InspectCall


def prepare_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image.convert("RGB")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(call: InspectCall) -> ImageInfo:
    path = Path(call.path)
    sha256 = _sha256(path)
    with Image.open(path) as img:
        width, height = img.size
        fmt = (img.format or "").upper()
        mode = img.mode
        phash = str(imagehash.phash(img))
    return ImageInfo(
        format=fmt,
        mode=mode,
        width=width,
        height=height,
        sha256=sha256,
        phash=phash,
    )
