from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

_BASELINE_RECALL = 0.968
_MAX_RECALL_DROP = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the live search HTTP interface")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cases = json.loads(args.fixture.read_text(encoding="utf-8"))
    headers = {}
    api_key = os.environ.get("HTTP_API_KEY_READONLY")
    if api_key:
        headers["X-API-Key"] = api_key

    report: dict[str, dict[str, float | int | str | None]] = {}
    with httpx.Client(
        base_url=args.base_url.rstrip("/"), timeout=args.timeout, headers=headers
    ) as client:
        for mode in ("image", "hybrid"):
            hits = 0
            reciprocal_ranks = 0.0
            version: str | None = None
            for case in cases:
                params = {"q": case["query"], "limit": 10}
                if mode == "hybrid":
                    params["mode"] = "hybrid"
                response = client.get("/search", params=params)
                response.raise_for_status()
                payload = response.json()
                current_version = payload.get("index_version")
                if version is None:
                    version = current_version
                elif current_version != version:
                    raise SystemExit(
                        f"active index changed during eval: {version!r} -> {current_version!r}"
                    )
                ids = [entry["id"] for entry in payload["results"]]
                try:
                    rank = ids.index(case["expected_image_id"]) + 1
                except ValueError:
                    continue
                hits += 1
                reciprocal_ranks += 1 / rank

            recall = hits / len(cases)
            report[mode] = {
                "queries": len(cases),
                "hits_at_10": hits,
                "recall_at_10": round(recall, 4),
                "mrr_at_10": round(reciprocal_ranks / len(cases), 4),
                "index_version": version,
            }

    print(json.dumps(report, indent=2, sort_keys=True))
    minimum = _BASELINE_RECALL - _MAX_RECALL_DROP
    failed = [mode for mode, result in report.items() if result["recall_at_10"] < minimum]
    if failed:
        raise SystemExit(
            f"recall gate failed for {', '.join(failed)}: minimum {minimum:.3f} "
            f"from baseline {_BASELINE_RECALL:.3f}"
        )


if __name__ == "__main__":
    main()
