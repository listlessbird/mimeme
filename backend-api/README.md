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

To use Modal inference instead of a compute gateway:

```bash
uv run modal setup
just modal-deploy
```

Set `COMPUTE_GPU_BACKEND=modal` and deploy the matching Modal app before
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

### Dedicated GPU inference gateway

The GPU machine needs NVIDIA drivers, NVIDIA Container Toolkit, Docker Compose,
and Tailscale. Port 8010 is intentionally bound only to its Tailscale address;
the compute gateway has no public authentication.

```bash
cp .env.gpu.example .env.gpu
# Fill in IMAGE_TAG, GPU_GATEWAY_BIND_IP, storage credentials, and HF_TOKEN.
docker compose --env-file .env.gpu -f docker.compose.gpu.yml config --quiet
docker compose --env-file .env.gpu -f docker.compose.gpu.yml pull
docker compose --env-file .env.gpu -f docker.compose.gpu.yml up -d
curl "http://$(grep '^GPU_GATEWAY_BIND_IP=' .env.gpu | cut -d= -f2):8010/v1/roles/inference/ready"
```

The Pi API and worker continue using the local compute service for image,
search, and index work, while inference goes to the Tailscale host:

```env
COMPUTE_GPU_BACKEND=local
COMPUTE_GATEWAY_URL=http://compute:8010
COMPUTE_INFERENCE_GATEWAY_URL=http://mimeme-gpu:8010
```

Use a Tailscale ACL that permits the Pi to reach the GPU node on TCP 8010. The
GPU node must also be able to reach both configured object-storage endpoints.
