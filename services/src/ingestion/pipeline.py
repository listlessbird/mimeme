import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List


from .config import CFG
from .hashing import compute_sha256, compute_phash
from .imaging import image_info
from .records import ImageRecord


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in CFG.exts


def walk_images(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file() and _is_supported(p):
            yield p


def process_path(path: Path, base: Path) -> ImageRecord | None:
    try:
        sha = compute_sha256(path)
    except Exception:
        return None

    w, h, fmt = image_info(path)

    ph = compute_phash(path)

    rel = path.relative_to(base).as_posix()

    return ImageRecord(sha, Path(rel), w, h, fmt, ph)


def process_paths_parallel(
    paths: Iterator[Path], base: Path, workers: int
) -> Iterator[ImageRecord]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_path, p, base): p for p in paths}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                yield rec


def batched(iterable: Iterable[ImageRecord], n: int) -> Iterator[list[ImageRecord]]:
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
