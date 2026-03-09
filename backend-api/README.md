# dev commands

docker compose up -d postgres redis minio temporal temporal-ui


uv run alembic upgrade head

after updating models,

alembic revision --autogenerate -m "drop unique on images.dataset"

apply it by

alembic revision --autogenerate -m "drop unique on images.dataset"


uv run ty check src

## Start infrastructure 
docker compose up -d postgres redis minio

(can either run temporal on docker or better yet)

temporal server start-dev

the above command starts both server and web ui

## start workers
docker compose up -d cpu-worker

or

uv run python -m workers.cpu_worker

docker compose --profile gpu up -d gpu-worker

uv run python -m workers.gpu_worker

## api
uv run uvicorn api.main:app --reload

# modal setup
uv run modal setup
uv run modal secret create mimeme-s3 \
  S3_ENDPOINT_URL=https://your-s3-endpoint \
  S3_REGION=us-east-1 \
  S3_ACCESS_KEY_ID=your-key \
  S3_SECRET_ACCESS_KEY=your-secret \
  S3_BUCKET=mimeme \
  S3_FORCE_PATH_STYLE=false


# apply migrations on prod
DB_URL=$(terraform -chdir=../infra output -raw neon_connection_uri | sed 's|^postgres://|postgresql://|') uv run alembic upgrade head
