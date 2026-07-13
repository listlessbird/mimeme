#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DB_URL="${DB_URL:-postgresql://postgres:postgres@localhost:5432/mimeme}"
PG_CONTAINER="${PG_CONTAINER:-mimeme-postgres}"
API_PORT="${API_PORT:-8000}"
DURATION="${DURATION:-30s}"
CONCURRENCY="${CONCURRENCY:-200}"
OVERLOAD_CONCURRENCY="${OVERLOAD_CONCURRENCY:-400}"
SEED_COUNT="${SEED_COUNT:-5000}"
OHA_BIN="${OHA_BIN:-oha}"
OUT_DIR="${OUT_DIR:-data/loadprobe}"
OVERLOAD_CPUS="${OVERLOAD_CPUS:-0.1}"

TARGET="http://127.0.0.1:${API_PORT}/images?limit=20"
API_LOG="${OUT_DIR}/api.log"

mkdir -p "$OUT_DIR"

if ! docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  docker compose up -d postgres
  sleep 3
fi

export DB_URL
SEED_COUNT="$SEED_COUNT" uv run python - <<'PY'
import os

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from shared.config import settings
from shared.models.orm import Base, Image

seed_count = int(os.environ["SEED_COUNT"])
engine = create_engine(settings.db_url_str)
Base.metadata.create_all(engine)
with Session(engine) as session:
    existing = session.scalar(select(func.count()).select_from(Image)) or 0
    for i in range(existing, seed_count):
        session.add(
            Image(
                sha256=f"loadprobe-{i:056d}",
                dataset="loadprobe",
                s3_key=f"images/loadprobe/{i}.jpg",
                width=640,
                height=480,
                format="jpeg",
                file_size=12345,
            )
        )
    session.commit()
    total = session.scalar(select(func.count()).select_from(Image)) or 0
print(f"images in db: {total}")
PY

APP_ENV=development DEBUG=false LOG_LEVEL=INFO \
RATE_LIMIT_ENABLED=false PRELOAD_TEXT_ENCODER_ON_STARTUP=false GPU_BACKEND=local \
AXIOM_API_TOKEN= AXIOM_DATASET= S3_ENDPOINT_URL=http://127.0.0.1:9 \
uv run uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" --log-level warning \
  >"$API_LOG" 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true; docker update --cpus 0 "$PG_CONTAINER" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 60); do
  curl -fsS -o /dev/null "$TARGET" 2>/dev/null && break
  sleep 1
done
curl -fsS -o /dev/null "$TARGET"

echo "== warmup, c=50 for 5s"
"$OHA_BIN" -z 5s -c 50 --no-tui --output-format quiet "$TARGET" >/dev/null
LAG_AFTER_WARMUP=$(grep -c event_loop_lag "$API_LOG" || true)

echo "== phase (a): normal load, c=${CONCURRENCY} for ${DURATION}"
"$OHA_BIN" -z "$DURATION" -c "$CONCURRENCY" --no-tui --output-format json "$TARGET" >"$OUT_DIR/normal.json"
LAG_AFTER_NORMAL=$(grep -c event_loop_lag "$API_LOG" || true)

echo "== phase (b): overload, c=${OVERLOAD_CONCURRENCY}, postgres constrained to ${OVERLOAD_CPUS} cpus"
docker update --cpus "$OVERLOAD_CPUS" "$PG_CONTAINER" >/dev/null
"$OHA_BIN" -z "$DURATION" -c "$OVERLOAD_CONCURRENCY" --no-tui --output-format json "$TARGET" >"$OUT_DIR/overload.json"
docker update --cpus 0 "$PG_CONTAINER" >/dev/null

kill "$API_PID"
wait "$API_PID" 2>/dev/null || true

LAG_TOTAL=$(grep -c event_loop_lag "$API_LOG" || true)
TRACEBACKS=$(grep -c Traceback "$API_LOG" || true)

LAG_AFTER_WARMUP="$LAG_AFTER_WARMUP" \
LAG_AFTER_NORMAL="$LAG_AFTER_NORMAL" LAG_TOTAL="$LAG_TOTAL" TRACEBACKS="$TRACEBACKS" \
OUT_DIR="$OUT_DIR" uv run python - <<'PY'
import json
import os

out_dir = os.environ["OUT_DIR"]
for phase in ("normal", "overload"):
    with open(f"{out_dir}/{phase}.json") as f:
        report = json.load(f)
    summary = report["summary"]
    percentiles = report["latencyPercentiles"]
    print(f"-- {phase}")
    print(f"   rps={summary['requestsPerSec']:.1f} total={report['statusCodeDistribution']}")
    for key in ("p50", "p90", "p99"):
        print(f"   {key}={percentiles[key] * 1000:.1f}ms")
    print(f"   errors={report['errorDistribution']}")
warmup = int(os.environ["LAG_AFTER_WARMUP"])
after_normal = int(os.environ["LAG_AFTER_NORMAL"])
total = int(os.environ["LAG_TOTAL"])
print(f"-- loop lag events: warmup={warmup} normal={after_normal - warmup} overload={total - after_normal}")
print(f"-- tracebacks in api log: {os.environ['TRACEBACKS']}")
PY

echo "raw reports in ${OUT_DIR}/, api log in ${API_LOG}"
