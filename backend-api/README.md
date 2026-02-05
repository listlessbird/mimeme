
# dev commands

docker compose up -d postgres redis minio temporal temporal-ui


uv run alembic upgrade head

# Start infrastructure only (for local dev)
docker compose up -d postgres redis minio temporal temporal-ui

# Start everything (no GPU)
docker compose up -d

# Start with GPU worker
docker compose --profile gpu up -d