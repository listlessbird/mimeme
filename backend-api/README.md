# Backend API

FastAPI + Temporal backend for meme ingestion, annotation, embedding, indexing, and search.

## Recommended Local Development

Run infrastructure in Docker, then run the API and Temporal worker locally with `uv`.

This is the most reliable local setup because the development compose file currently provides Postgres, Redis, MinIO, and optional Temporal services, while the API and worker are easier to run directly from the checked-out source.

### 1. Configure Environment

From `backend-api/`:

```bash
cp .env.example .env
```

The defaults in `.env.example` point at local Docker services:

- Postgres: `localhost:5432`
- Redis: `localhost:6379`
- MinIO S3: `localhost:9000`
- Temporal: `localhost:7233`

For local CPU-only embedding runs, set:

```bash
EMBED_DEVICE=cpu
```

The default `GPU_BACKEND=local` runs local model inference in the worker. To use Modal instead, complete the Modal setup below and set:

```bash
GPU_BACKEND=modal
```

### 2. Start Infra

Start Postgres, Redis, and MinIO:

```bash
docker compose up -d postgres redis minio
```

Start Temporal with the Temporal CLI:

```bash
temporal server start-dev
```

Keep this command running. It starts both the Temporal server and Temporal Web UI.

Do not also start the Docker `temporal` service when using `temporal server start-dev`; both bind to port `7233`.

### 3. Apply Migrations

In a second terminal:

```bash
cd backend-api
uv run alembic upgrade head
```

Create the test database once if you want backend tests to run against local Postgres:

```bash
docker compose exec postgres createdb -U postgres mimeme_test
```

The test suite looks for `TEST_DB_URL` first, then tries
`postgresql://postgres:postgres@localhost:5432/mimeme_test`. If that database
is not reachable, tests fall back to an in-memory SQLite database.

### 4. Start The API

```bash
cd backend-api
uv run uvicorn api.main:app --reload
```

The API will create the configured MinIO bucket on startup if it does not already exist.

### 5. Start The Worker

In a third terminal:

```bash
cd backend-api
uv run python -m workers.worker
```

The worker must be running for ingest and rebuild jobs to execute.

### 6. Local URLs

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/live`
- Temporal UI: `http://localhost:8233`
- MinIO console: `http://localhost:9001`

MinIO credentials:

```text
minioadmin / minioadmin
```

### 7. Manual Smoke Test

Check service health:

```bash
curl http://localhost:8000/live
```

Queue one image for ingestion:

```bash
curl -X POST http://localhost:8000/images \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://picsum.photos/seed/test-meme/512/512"],"dataset":"manual","tags":["manual"]}'
```

Check the returned job:

```bash
curl http://localhost:8000/jobs/<job_id>
```

After images are embedded, rebuild the search index:

```bash
curl -X POST http://localhost:8000/jobs/rebuild-index \
  -H 'Content-Type: application/json' \
  -d '{"force":true}'
```

Search once the rebuild job completes:

```bash
curl 'http://localhost:8000/search?q=cat&mode=hybrid'
```

In `APP_ENV=development`, API-key auth is bypassed for local manual testing.

### 8. Stop Services

Stop local foreground processes with `Ctrl-C`.

Stop Docker infra:

```bash
docker compose down
```

To also remove local Docker volumes:

```bash
docker compose down -v
```

## Tmux Dev Helper

The repo includes a tmux helper that starts Docker infra, Temporal CLI, API, and worker panes:

```bash
cd backend-api
./dev.sh
```

## Alternative Docker Temporal

If you prefer Docker Temporal instead of the Temporal CLI:

```bash
docker compose up -d postgres redis minio temporal temporal-ui
```

Then keep running the API and worker locally:

```bash
uv run alembic upgrade head
uv run uvicorn api.main:app --reload
uv run python -m workers.worker
```

Temporal UI is available at `http://localhost:8088` in this mode.

## Modal Setup

```bash
uv run modal setup
uv run modal secret create findmeme-s3 \
  S3_ENDPOINT_URL=https://your-s3-endpoint \
  S3_REGION=auto \
  S3_ACCESS_KEY_ID=your-key \
  S3_SECRET_ACCESS_KEY=your-secret \
  S3_BUCKET=findmeme-prod-storage \
  S3_FORCE_PATH_STYLE=true
```

## Development Commands

Install all backend development groups:

```bash
uv sync --all-groups
```

The repo includes a `justfile` for common backend commands if you have `just` installed:

```bash
just fmt
just lint
just type
just test
just check
```

The same commands can be run directly with `uv`:

Format code and sort imports:

```bash
uv run ruff check --select I --fix src tests
uv run ruff format src tests
```

Run lint, typecheck, and tests:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pyright
uv run pytest -q
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run tests:

```bash
uv run pytest -q
```

To force tests to use a specific Postgres database:

```bash
TEST_DB_URL=postgresql://postgres:postgres@localhost:5432/mimeme_test uv run pytest -q
```

Run Ruff:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
```

Create a migration after model changes:

```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

Apply migrations against the production database:

```bash
DB_URL=$(terraform -chdir=../infra output -raw neon_connection_uri | sed 's|^postgres://|postgresql://|') uv run alembic upgrade head
```
