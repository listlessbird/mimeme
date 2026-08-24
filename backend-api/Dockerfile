ARG RELEASE_ID=dev

FROM python:3.13-slim AS base

ARG RELEASE_ID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MIMEME_RELEASE_ID=${RELEASE_ID}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app


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

CMD ["sh", "-c", "alembic upgrade head && uvicorn mimeme.api.main:app --host 0.0.0.0 --port 8000"]

FROM builder AS compute-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra compute --extra local-gpu

FROM base AS compute

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libvips42 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=compute-builder --chown=appuser:appuser /app/.venv /app/.venv

RUN mkdir -p /app/.cache/huggingface /tmp/mimeme-compute && \
    chown appuser:appuser /app/.cache/huggingface /tmp/mimeme-compute

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    HUGGINGFACE_HUB_CACHE="/app/.cache/huggingface/hub" \
    COMPUTE_BIND_HOST=0.0.0.0 \
    COMPUTE_BIND_PORT=8010 \
    COMPUTE_SOCKET_DIR=/tmp/mimeme-compute

USER appuser

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8010/ready || exit 1

CMD ["python", "-m", "mimeme.compute"]

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

CMD ["python", "-m", "mimeme.worker"]
