## Local development

From `backend-api/`:

```bash
cp .env.example .env
uv sync --all-groups --all-extras
docker compose up -d postgres minio temporal temporal-ui
uv run alembic upgrade head
```

Start the three application processes in separate terminals:

```bash
uv run api
uv run worker
uv run compute
```

The default URLs are API `http://localhost:8000`, Temporal UI
`http://localhost:8088`, and MinIO console `http://localhost:9001`. Development
mode bypasses API-key auth. Production requests use `X-API-Key`.

To use Modal inference instead of local inference:

```bash
uv run modal setup
just modal-deploy
```

Set `COMPUTE__GPU_BACKEND=modal` and deploy the matching Modal app before
starting API/worker readiness checks.

## Verification

```bash
DEBUG=false just check
uv sync --frozen --all-groups --all-extras
uv build
uv run alembic upgrade head
uv run alembic current
uv run alembic check
docker compose config
docker compose build api worker compute
```
`just test-model` tests models

## Production

CI publishes API, worker, and compute artifacts tagged by release. Deploy one
matching release ID across Modal, compute, API, and worker. Do not deploy a new
worker onto an old task queue or expose the API before migrations, schedule
reconciliation, and readiness pass.
