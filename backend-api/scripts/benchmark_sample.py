#!/usr/bin/env python3
"""Download a reproducible random image sample from configured media storage."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from mimeme import storage
from mimeme.config import Settings

MAX_IMAGE_BYTES = 64 * 1024 * 1024


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/sample"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--prefix", default="images/")
    parser.add_argument("--max-scan", type=int, default=0, help="0 scans the full prefix")
    return parser.parse_args()


def _config(settings: Settings) -> storage.Config:
    media = settings.media
    return storage.Config(
        endpoint_url=media.s3_endpoint_url,
        region=media.s3_region,
        access_key=media.s3_access_key_id,
        secret_key=media.s3_secret_access_key,
        bucket=media.s3_bucket,
        force_path_style=media.s3_force_path_style,
    )


async def _main() -> None:
    args = _args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty: {args.output_dir}")

    settings = Settings()
    rng = random.Random(args.seed)
    store = await storage.S3.open(_config(settings))
    candidates: list[storage.Info] = []
    seen = 0
    try:
        async for info in store.list(prefix=args.prefix):
            if info.length <= 0 or info.length > MAX_IMAGE_BYTES:
                continue
            seen += 1
            if len(candidates) < args.count * 4:
                candidates.append(info)
            else:
                replacement = rng.randrange(seen)
                if replacement < len(candidates):
                    candidates[replacement] = info
            if args.max_scan and seen >= args.max_scan:
                break

        rng.shuffle(candidates)
        manifest: list[dict[str, object]] = []
        for info in candidates:
            if len(manifest) >= args.count:
                break
            try:
                data = await store.read_bytes(info.object, max_bytes=MAX_IMAGE_BYTES)
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                    image_format = (image.format or "image").lower()
                suffix = ".jpg" if image_format == "jpeg" else f".{image_format}"
                path = args.output_dir / f"{len(manifest):04d}{suffix}"
                path.write_bytes(data)
                public_url = (
                    f"{settings.media.public_base_url.rstrip('/')}/{quote(info.object.key)}"
                )
                manifest.append(
                    {
                        "index": len(manifest),
                        "path": str(path.resolve()),
                        "media_key": info.object.key,
                        "public_url": public_url,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            except Exception as exc:
                print(f"skip {info.object.key}: {type(exc).__name__}: {exc}")
    finally:
        await store.close()

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"downloaded": len(manifest), "scanned": seen, "manifest": str(manifest_path)})
    )
    if len(manifest) < args.count:
        raise SystemExit(f"only found {len(manifest)} valid images; requested {args.count}")


if __name__ == "__main__":
    asyncio.run(_main())
