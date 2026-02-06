#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import httpx

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
TERMINAL_JOB_STATUSES = {"COMPLETED", "FAILED", "CANCELED", "CANCELLED"}


@dataclass
class IngestResult:
    file_path: Path
    dataset: str
    job_id: str | None
    status: str
    image_id: int | None
    caption: str | None
    ocr_text: str | None
    error: str | None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _local_file_server(root_dir: Path, port: int):
    handler = partial(SimpleHTTPRequestHandler, directory=str(root_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _collect_first_images(raw_dir: Path, limit: int) -> list[Path]:
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files[:limit]


def _compact(text: str | None, max_chars: int = 120) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "."}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "image"


def _wait_for_job(client: httpx.Client, api_base: str, job_id: str, timeout_sec: int) -> dict:
    started = time.monotonic()
    while True:
        resp = client.get(f"{api_base}/jobs/{job_id}", timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        status = str(payload.get("status", "")).upper()
        if status in TERMINAL_JOB_STATUSES:
            return payload
        if (time.monotonic() - started) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for job {job_id}")
        time.sleep(2)


def _fetch_single_image_for_dataset(client: httpx.Client, api_base: str, dataset: str) -> dict | None:
    resp = client.get(
        f"{api_base}/images",
        params={"dataset": dataset, "limit": 1, "offset": 0, "sort": "newest"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    images = payload.get("images", [])
    return images[0] if images else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local memes through backend API.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--raw-dir", default="data/raw_memes")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--tags", default="local,raw-memes")
    parser.add_argument("--dataset-prefix", default="raw-memes")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.exists():
        print(f"[error] raw dir not found: {raw_dir}")
        return 1

    files = _collect_first_images(raw_dir, args.limit)
    if not files:
        print(f"[error] no image files found under: {raw_dir}")
        return 1

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    run_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    port = _find_free_port()
    api_base = args.api_base.rstrip("/")

    print(f"[info] serving local files from {raw_dir} on http://127.0.0.1:{port}")
    print(f"[info] ingesting {len(files)} images via {api_base}/images")

    results: list[IngestResult] = []
    with _local_file_server(raw_dir, port), httpx.Client() as client:
        for idx, file_path in enumerate(files, start=1):
            rel = file_path.relative_to(raw_dir)
            rel_url = "/".join(quote(part) for part in rel.parts)
            image_url = f"http://127.0.0.1:{port}/{rel_url}"
            dataset = f"{args.dataset_prefix}-{run_stamp}-{idx:03d}-{_slug(file_path.stem)[:30]}"

            print(f"\n[{idx}/{len(files)}] queue {rel} -> dataset={dataset}")
            try:
                enqueue_resp = client.post(
                    f"{api_base}/images",
                    json={"urls": [image_url], "dataset": dataset, "tags": tags},
                    timeout=30,
                )
                enqueue_resp.raise_for_status()
                enqueue = enqueue_resp.json()
                job_id = enqueue.get("job_id")
                if not job_id:
                    raise RuntimeError(f"Missing job_id in response: {enqueue}")
                print(f"  [queued] job_id={job_id}")

                job_payload = _wait_for_job(client, api_base, job_id, args.timeout_sec)
                job_status = str(job_payload.get("status", "UNKNOWN"))
                print(f"  [job] status={job_status}")

                if job_status == "COMPLETED":
                    image = _fetch_single_image_for_dataset(client, api_base, dataset)
                    if image:
                        image_id = image.get("id")
                        caption = image.get("caption")
                        ocr_text = image.get("ocr_text")
                        print(
                            "  [ok] "
                            f"image_id={image_id} "
                            f"caption={_compact(caption)} "
                            f"ocr={_compact(ocr_text)}"
                        )
                        results.append(
                            IngestResult(
                                file_path=file_path,
                                dataset=dataset,
                                job_id=job_id,
                                status="COMPLETED",
                                image_id=int(image_id) if image_id is not None else None,
                                caption=caption,
                                ocr_text=ocr_text,
                                error=None,
                            )
                        )
                    else:
                        err = "Job completed but no image found by dataset"
                        print(f"  [warn] {err}")
                        results.append(
                            IngestResult(
                                file_path=file_path,
                                dataset=dataset,
                                job_id=job_id,
                                status="COMPLETED",
                                image_id=None,
                                caption=None,
                                ocr_text=None,
                                error=err,
                            )
                        )
                else:
                    err = (
                        f"job status={job_status} "
                        f"message={job_payload.get('message')} "
                        f"result={job_payload.get('result')}"
                    )
                    print(f"  [fail] {err}")
                    results.append(
                        IngestResult(
                            file_path=file_path,
                            dataset=dataset,
                            job_id=job_id,
                            status=job_status,
                            image_id=None,
                            caption=None,
                            ocr_text=None,
                            error=err,
                        )
                    )
            except Exception as exc:
                print(f"  [error] {type(exc).__name__}: {exc}")
                results.append(
                    IngestResult(
                        file_path=file_path,
                        dataset=dataset,
                        job_id=None,
                        status="ERROR",
                        image_id=None,
                        caption=None,
                        ocr_text=None,
                        error=str(exc),
                    )
                )

    success = sum(1 for r in results if r.status == "COMPLETED" and r.image_id is not None)
    failed = len(results) - success
    print("\n[summary]")
    print(f"  total={len(results)} success={success} failed={failed}")
    for r in results:
        if r.status != "COMPLETED" or r.image_id is None:
            print(f"  - file={r.file_path.name} status={r.status} error={_compact(r.error, 180)}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
