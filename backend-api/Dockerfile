FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app


COPY vendored ./vendored

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM base AS api

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser alembic.ini ./

RUN mkdir -p /app/data /app/.cache/huggingface && \
    chown appuser:appuser /app/data /app/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    HUGGINGFACE_HUB_CACHE="/app/.cache/huggingface/hub"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]

FROM base AS worker

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

RUN mkdir -p /app/data /app/.cache/huggingface && \
    chown appuser:appuser /app/data /app/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    HUGGINGFACE_HUB_CACHE="/app/.cache/huggingface/hub"

USER appuser

CMD ["python", "-m", "workers.worker"]
