#!/usr/bin/env python3
"""Select a balanced GIF eval set and build contact sheets for annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps, ImageSequence

SEED = "mimeme-gif-annotation-v1"
HOLDOUT_FRACTION = 0.2
SAMPLE_COUNT = 8
CONTACT_COLUMNS = 4
TILE_SIZE = (320, 240)


@dataclass(frozen=True)
class Candidate:
    sha256: str
    search_term: str
    width: int
    height: int
    n_frames: int
    duration_ms: int
    n_bytes: int

    @property
    def frame_bucket(self) -> str:
        if self.n_frames <= 8:
            return "short"
        if self.n_frames >= 50:
            return "long"
        return "medium"


def stable_rank(sha256: str) -> str:
    return hashlib.sha256(f"{SEED}:{sha256}".encode()).hexdigest()


def holdout_split(sha256: str) -> str:
    digest = hashlib.sha256(f"holdout:{sha256}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "holdout" if bucket < HOLDOUT_FRACTION else "tune"


def load_candidates(manifest_path: Path) -> list[Candidate]:
    raw = json.loads(manifest_path.read_text())
    return [
        Candidate(
            sha256=entry["sha256"],
            search_term=entry["search_term"],
            width=entry["width"],
            height=entry["height"],
            n_frames=entry["n_frames"],
            duration_ms=entry["duration_ms"],
            n_bytes=entry["n_bytes"],
        )
        for entry in raw.values()
    ]


def select_candidates(candidates: list[Candidate], count: int) -> list[Candidate]:
    by_term: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_term[candidate.search_term].append(candidate)

    selected: list[Candidate] = []
    selected_shas: set[str] = set()
    term_counts: Counter[str] = Counter()

    for term in sorted(by_term):
        ranked = sorted(by_term[term], key=lambda item: stable_rank(item.sha256))
        first = ranked[0]
        second = next(
            (item for item in ranked[1:] if item.frame_bucket != first.frame_bucket),
            ranked[1] if len(ranked) > 1 else first,
        )
        for item in (first, second):
            if item.sha256 in selected_shas or len(selected) >= count:
                continue
            selected.append(item)
            selected_shas.add(item.sha256)
            term_counts[item.search_term] += 1

    targets = {"short": round(count * 0.15), "medium": round(count * 0.5)}
    targets["long"] = count - targets["short"] - targets["medium"]
    bucket_counts = Counter(item.frame_bucket for item in selected)
    remaining = [item for item in candidates if item.sha256 not in selected_shas]

    while len(selected) < count and remaining:
        deficits = {bucket: targets[bucket] - bucket_counts[bucket] for bucket in targets}
        desired_bucket = max(deficits, key=lambda bucket: (deficits[bucket], bucket))
        pool = [item for item in remaining if item.frame_bucket == desired_bucket]
        if not pool:
            pool = remaining
        item = min(
            pool,
            key=lambda candidate: (
                term_counts[candidate.search_term],
                stable_rank(candidate.sha256),
            ),
        )
        selected.append(item)
        selected_shas.add(item.sha256)
        term_counts[item.search_term] += 1
        bucket_counts[item.frame_bucket] += 1
        remaining.remove(item)

    return sorted(selected, key=lambda item: stable_rank(item.sha256))


def decode_frames(gif_path: Path) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    with Image.open(gif_path) as image:
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGBA").copy())
            durations.append(max(int(frame.info.get("duration", 0) or 0), 10))
    return frames, durations


def sampled_indices(durations: list[int], sample_count: int = SAMPLE_COUNT) -> list[int]:
    total_duration = sum(durations)
    targets = [(index + 0.5) * total_duration / sample_count for index in range(sample_count)]
    indices: list[int] = []
    elapsed = 0
    target_index = 0
    for frame_index, duration in enumerate(durations):
        elapsed += duration
        while target_index < len(targets) and targets[target_index] <= elapsed:
            indices.append(frame_index)
            target_index += 1
    while len(indices) < sample_count:
        indices.append(len(durations) - 1)
    return indices


def distinct_samples(
    frames: list[Image.Image], durations: list[int]
) -> list[tuple[int, int, Image.Image]]:
    timeline: list[int] = []
    elapsed = 0
    for duration in durations:
        timeline.append(elapsed)
        elapsed += duration

    samples: list[tuple[int, int, Image.Image]] = []
    hashes: list[imagehash.ImageHash] = []
    for frame_index in sampled_indices(durations):
        frame = frames[frame_index].convert("RGB")
        frame_hash = imagehash.phash(frame)
        if hashes and any(frame_hash - existing <= 2 for existing in hashes):
            continue
        hashes.append(frame_hash)
        samples.append((frame_index, timeline[frame_index], frame))
    if not samples:
        samples.append((0, 0, frames[0].convert("RGB")))
    return samples


def build_contact_sheet(samples: list[tuple[int, int, Image.Image]], output_path: Path) -> None:
    rows = (len(samples) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    label_height = 26
    sheet = Image.new(
        "RGB",
        (CONTACT_COLUMNS * TILE_SIZE[0], rows * (TILE_SIZE[1] + label_height)),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    for sample_number, (frame_index, timestamp_ms, frame) in enumerate(samples, start=1):
        column = (sample_number - 1) % CONTACT_COLUMNS
        row = (sample_number - 1) // CONTACT_COLUMNS
        x = column * TILE_SIZE[0]
        y = row * (TILE_SIZE[1] + label_height)
        fitted = ImageOps.contain(frame, TILE_SIZE)
        tile = Image.new("RGB", TILE_SIZE, "#000000")
        tile.paste(
            fitted,
            ((TILE_SIZE[0] - fitted.width) // 2, (TILE_SIZE[1] - fitted.height) // 2),
        )
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 8, y + TILE_SIZE[1] + 7),
            f"sample {sample_number} | frame {frame_index} | {timestamp_ms} ms",
            fill="#f4f4f5",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def prepare(args: argparse.Namespace) -> None:
    candidates = load_candidates(args.manifest)
    selected = select_candidates(candidates, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    contact_dir = args.output / "contact-sheets"

    items = []
    for position, candidate in enumerate(selected, start=1):
        gif_path = args.gifs / f"{candidate.sha256}.gif"
        frames, durations = decode_frames(gif_path)
        samples = distinct_samples(frames, durations)
        contact_path = contact_dir / f"{candidate.sha256}.jpg"
        build_contact_sheet(samples, contact_path)
        items.append(
            {
                "sha256": candidate.sha256,
                "position": position,
                "asset_key": f"gif-annotation/v1/gifs/{candidate.sha256}.gif",
                "contact_sheet": str(contact_path.relative_to(args.output)),
                "contact_sheet_asset_key": (
                    f"gif-annotation/v1/contact-sheets/{candidate.sha256}.jpg"
                ),
                "split": holdout_split(candidate.sha256),
                "width": candidate.width,
                "height": candidate.height,
                "n_frames": candidate.n_frames,
                "duration_ms": candidate.duration_ms,
                "n_bytes": candidate.n_bytes,
                "sampled_frames": [
                    {"sample": number, "frame": frame_index, "timestamp_ms": timestamp_ms}
                    for number, (frame_index, timestamp_ms, _) in enumerate(samples, start=1)
                ],
            }
        )

    dataset = {
        "version": 1,
        "selection_seed": SEED,
        "n_gifs": len(items),
        "items": items,
    }
    (args.output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    bucket_counts = Counter(candidate.frame_bucket for candidate in selected)
    total_bytes = sum(candidate.n_bytes for candidate in selected)
    print(
        f"prepared {len(items)} gifs ({total_bytes / 1024 / 1024:.1f} MiB) "
        f"with frame buckets {dict(bucket_counts)}"
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "mimeme-gif-experiment"
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--manifest", type=Path, default=experiment / "data/manifest.json")
    parser.add_argument("--gifs", type=Path, default=experiment / "data/gifs")
    parser.add_argument("--output", type=Path, default=experiment / "data/gif-annotation-v1")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
