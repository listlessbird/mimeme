from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def leaks_ocr(query: str, ocr_text: str, max_ngram: int = 4) -> bool:
    query_words = words(query)
    ocr = " ".join(words(ocr_text))
    if len(query_words) <= max_ngram:
        return False
    return any(
        " ".join(query_words[i : i + max_ngram + 1]) in ocr
        for i in range(len(query_words) - max_ngram)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evals/eval_queries_input.jsonl"))
    parser.add_argument("--llm-outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("evals/search_eval_set.json"))
    args = parser.parse_args()

    records = {}
    with args.input.open() as f:
        for line in f:
            record = json.loads(line)
            records[record["image_id"]] = record

    raw: list[tuple[str, int]] = []
    for path in args.llm_outputs:
        for item in json.loads(path.read_text())["items"]:
            image_id = item["image_id"]
            if image_id not in records:
                raise ValueError(f"{path}: unknown image_id {image_id}")
            for query in item["queries"]:
                query = " ".join(query.strip().lower().split())
                if query:
                    raw.append((query, image_id))

    dropped: dict[str, list[str]] = {"duplicate": [], "ocr_leak": [], "too_generic": []}
    counts = Counter(q for q, _ in raw)
    seen: set[str] = set()
    kept: list[dict] = []
    for query, image_id in raw:
        if counts[query] > 1:
            if query not in seen:
                dropped["duplicate"].append(query)
            seen.add(query)
            continue
        if leaks_ocr(query, records[image_id]["ocr_text"]):
            dropped["ocr_leak"].append(query)
            continue
        if len(words(query)) < 2:
            dropped["too_generic"].append(query)
            continue
        kept.append({"query": query, "expected_image_id": image_id, "origin": "llm"})

    kept.sort(key=lambda e: (e["expected_image_id"], e["query"]))
    args.out.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n")

    covered = len({e["expected_image_id"] for e in kept})
    print(f"kept {len(kept)} queries covering {covered}/{len(records)} images → {args.out}")
    for reason, queries in dropped.items():
        print(f"dropped {reason}: {len(queries)}")
        for query in queries[:10]:
            print(f"  - {query}")


if __name__ == "__main__":
    main()
