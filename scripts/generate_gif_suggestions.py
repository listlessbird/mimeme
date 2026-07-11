#!/usr/bin/env python3
"""Generate resumable structured GIF annotations through Codex CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MODEL = "gpt-5.4-mini"
PROMPT_VERSION = 2
PROMPT = """You are annotating a meme GIF for a semantic retrieval evaluation.
The attached contact sheet contains time-ordered, deduplicated samples from one GIF. The labels
under each tile identify sample numbers; use those sample numbers in supporting_frame_numbers.
Only cite the numbered contact-sheet samples (1 through the number of tiles), never original GIF
frame indices or timestamps.

Return only the requested JSON. Follow these rules:
- Transcribe visible text exactly. Do not invent obscured or unreadable words.
- Describe concrete people, objects, actions, expressions, and scene changes.
- visual queries describe visible content without relying on overlaid text.
- caption queries use exact or near-exact visible text and may be empty when no text is visible.
- natural queries are short phrases a person might genuinely type, including identity-aware searches
  such as "SpongeBob confused", "Michael Scott no", or "Jim Carrey typing" when supported.
- Actively infer recognizable actors, public figures, characters, shows, movies, meme templates, and
  commonly known scenes from the frames, and include useful names in natural queries and descriptions.
- Calibrate uncertainty: use likely identity phrasing for plausible recognition instead of withholding
  it, but do not confidently fabricate a name from weak visual evidence.
- Do not name unknown or private people; describe them generically.
- Never use filenames, collection provenance, or other hidden hints. Base every claim on the frames.
- State remaining uncertainty briefly instead of guessing.
"""


def run_codex(
    *, sha256: str, contact_sheet: Path, output_path: Path, schema: Path, model: str, cwd: Path
) -> tuple[str, str]:
    if output_path.exists():
        return sha256, "cached"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "codex",
        "exec",
        "--model",
        model,
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--color",
        "never",
        "--image",
        str(contact_sheet),
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(output_path),
        PROMPT,
    ]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        error = result.stderr.strip() or result.stdout.strip() or "codex failed"
        raise RuntimeError(f"{sha256}: {error[-1200:]}")
    json.loads(output_path.read_text())
    return sha256, "generated"


def generate(args: argparse.Namespace) -> None:
    dataset = json.loads((args.data / "dataset.json").read_text())
    suggestion_dir = args.data / f"suggestions-v{PROMPT_VERSION}"
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for item in dataset["items"]:
            sha256 = item["sha256"]
            jobs.append(
                executor.submit(
                    run_codex,
                    sha256=sha256,
                    contact_sheet=args.data / item["contact_sheet"],
                    output_path=suggestion_dir / f"{sha256}.json",
                    schema=args.schema,
                    model=args.model,
                    cwd=args.data,
                )
            )
        completed = 0
        for future in as_completed(jobs):
            sha256, status = future.result()
            completed += 1
            print(f"[{completed}/{len(jobs)}] {sha256[:10]} {status}")

    suggestions = {
        "version": PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "items": {
            item["sha256"]: json.loads(
                (suggestion_dir / f"{item['sha256']}.json").read_text()
            )
            for item in dataset["items"]
        },
    }
    (args.data / "suggestions.json").write_text(json.dumps(suggestions, indent=2) + "\n")
    print(f"wrote {len(suggestions['items'])} suggestions using {args.model}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "mimeme-gif-experiment/data/gif-annotation-v1",
    )
    parser.add_argument(
        "--schema", type=Path, default=root / "scripts/gif_suggestion.schema.json"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
