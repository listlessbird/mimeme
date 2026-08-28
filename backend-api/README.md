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
mode bypasses admin auth. Production web-admin requests use a signed GitHub
OAuth session. Scripts may continue to use `X-API-Key`.

## GitHub admin sign-in

Create one GitHub OAuth app with both callback URLs:

- `https://api.mimeme.dev/auth/github/callback`
- `http://localhost:8000/auth/github/callback`

Set `AUTH_GITHUB_CALLBACK_URL` to the matching URL in each environment. Configure
the remaining `AUTH_GITHUB_*`, `AUTH_ALLOWED_GITHUB_IDS`, `AUTH_SESSION_SECRET`,
`AUTH_COOKIE_DOMAIN`, and `AUTH_UI_URL` values shown in `.env.example`.

Find the durable numeric ID for the currently authenticated GitHub account with:

```bash
gh api user --jq .id
```

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
# Fill in IMAGE_TAG, GPU_GATEWAY_BIND_IP, storage/Axiom credentials, and HF_TOKEN.
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

## GPU benchmarks and cost planning

Use a reproducible sample from the configured media bucket, then measure the
model, compute gateway, and full Temporal path separately:

```bash
uv run python scripts/benchmark_sample.py --count 100
uv run python scripts/benchmark_models.py \
  --manifest data/benchmarks/sample/manifest.json --config B \
  --output data/benchmarks/results/models-B.json
uv run python scripts/benchmark_compute_gateway.py \
  --manifest data/benchmarks/sample/manifest.json --batch-size 2 \
  --output data/benchmarks/results/gateway.json
uv run python scripts/benchmark_temporal_ingestion.py \
  --manifest data/benchmarks/sample/manifest.json --limit 100 \
  --worker-log data/benchmarks/worker.jsonl --gpu-price-per-hour 0.20 \
  --output data/benchmarks/results/temporal.json
```

The RTX 5060 benchmark established the dedicated-GPU defaults: keep both models
resident, leave vision compilation disabled, and use ingestion fanout 2. Caption
generation is the throughput bottleneck; larger SigLIP batches did not improve
end-to-end throughput materially.

Convert a full-ingestion result into monthly serverless and always-on estimates:

```bash
uv run python scripts/benchmark_cost.py \
  --result data/benchmarks/results/fanout-2.json \
  --gpu-price-per-hour 0.20 \
  --cold-start-seconds 14.15 --runs-per-month 30
```

`processing_cost_per_1000` is the warm variable cost. `serverless_gpu_cost`
adds the specified cold start once per run. `always_on_gpu_cost` is simply the
hourly GPU price times 730 hours, so throughput only changes its utilization,
not its monthly bill. Provider CPU, RAM, storage, network, and minimum billing
charges are intentionally excluded; add them when comparing an actual offer.
