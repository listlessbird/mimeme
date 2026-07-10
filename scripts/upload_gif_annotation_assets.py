#!/usr/bin/env python3
"""Upload the selected GIF annotation dataset to its dedicated R2 prefix."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BUCKET = "mimeme-gif-annotation"
PREFIX = "gif-annotation/v1"


@dataclass(frozen=True)
class Upload:
    key: str
    path: Path
    content_type: str
    cache_control: str


def upload_object(wrangler: Path, upload: Upload, attempts: int = 4) -> str:
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            [
                str(wrangler),
                "r2",
                "object",
                "put",
                f"{BUCKET}/{upload.key}",
                "--file",
                str(upload.path),
                "--content-type",
                upload.content_type,
                "--cache-control",
                upload.cache_control,
                "--remote",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return upload.key
        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))

    error = result.stderr.strip() or result.stdout.strip() or "upload failed"
    raise RuntimeError(f"{upload.key}: {error[-1200:]}")


def build_uploads(args: argparse.Namespace) -> list[Upload]:
    dataset_path = args.data / "dataset.json"
    suggestions_path = args.data / "suggestions.json"
    dataset = json.loads(dataset_path.read_text())
    uploads = [
        Upload(PREFIX + "/dataset.json", dataset_path, "application/json", "no-cache"),
    ]
    if suggestions_path.exists():
        uploads.append(
            Upload(
                PREFIX + "/suggestions.json",
                suggestions_path,
                "application/json",
                "no-cache",
            )
        )
    elif not args.allow_missing_suggestions:
        raise SystemExit("suggestions.json is missing; finish generation or pass --allow-missing-suggestions")

    for item in dataset["items"]:
        sha256 = item["sha256"]
        uploads.extend(
            [
                Upload(
                    f"{PREFIX}/gifs/{sha256}.gif",
                    args.gifs / f"{sha256}.gif",
                    "image/gif",
                    "public, max-age=31536000, immutable",
                ),
                Upload(
                    f"{PREFIX}/contact-sheets/{sha256}.jpg",
                    args.data / "contact-sheets" / f"{sha256}.jpg",
                    "image/jpeg",
                    "public, max-age=31536000, immutable",
                ),
            ]
        )
    return uploads


def main(args: argparse.Namespace) -> None:
    uploads = build_uploads(args)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(upload_object, args.wrangler, upload) for upload in uploads]
        completed = 0
        for future in as_completed(futures):
            key = future.result()
            completed += 1
            print(f"[{completed}/{len(uploads)}] {key}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "mimeme-gif-experiment"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=experiment / "data/gif-annotation-v1"
    )
    parser.add_argument("--gifs", type=Path, default=experiment / "data/gifs")
    parser.add_argument(
        "--wrangler", type=Path, default=root / "webui/node_modules/.bin/wrangler"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-missing-suggestions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
