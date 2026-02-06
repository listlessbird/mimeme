
# dev commands

docker compose up -d postgres redis minio temporal temporal-ui


uv run alembic upgrade head

after updating models,

alembic revision --autogenerate -m "drop unique on images.dataset"

apply it by

alembic revision --autogenerate -m "drop unique on images.dataset"


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

