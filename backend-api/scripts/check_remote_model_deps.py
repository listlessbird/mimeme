#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import check_imports

MISSING_RE = re.compile(
    r"requires the following packages that were not found in your environment:\s*(.+?)\.",
    re.IGNORECASE | re.DOTALL,
)


def parse_missing_packages(message: str) -> list[str]:
    match = MISSING_RE.search(message)
    if not match:
        return []
    raw = match.group(1).replace("\n", " ").strip()
    parts = [part.strip().strip("`'\"") for part in raw.split(",")]
    return [part for part in parts if part]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check missing Python deps required by a remote-code Hugging Face model."
    )
    parser.add_argument("--model-id", default="vikhyatk/moondream2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default="/tmp/hf-model-cache")
    args = parser.parse_args()

    model_dir = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            allow_patterns=["*.py", "*.json"],
            cache_dir=args.cache_dir,
        )
    )

    missing: set[str] = set()
    for py_file in model_dir.rglob("*.py"):
        try:
            check_imports(py_file)
        except ImportError as exc:
            missing.update(parse_missing_packages(str(exc)))

    if missing:
        ordered = sorted(missing)
        print("Missing packages:")
        for pkg in ordered:
            print(f"- {pkg}")
        print()
        print("Install all at once:")
        print(f"uv add {' '.join(ordered)}")
        return 1

    print("No missing remote-model Python packages detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
