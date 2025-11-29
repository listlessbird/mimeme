from __future__ import annotations

from celery import Celery
from config import settings

celery_app = Celery("find_meme", broker=settings.broker_url, backend=settings.result_backend)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="utc",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    result_expires=60 * 60 * 24,
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.worker_concurrency,
    task_routes={
        "api.tasks.ingest.*": {"queue": "ingest"},
        "api.tasks.index.*": {"queue": "index"},
        "api.tasks.annotate.*": {"queue": "gpu"},
        "api.tasks.embed.*": {"queue": "gpu"},
    },
    task_default_queue="default",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    beat_schedule={
        "rebuild-index-daily": {
            "task": "api.tasks.index.rebuild_index_task",
            "schedule": 24 * 60 * 20,
            "kwargs": {"force": False},
        },
        "cleanup-old-indexes": {
            "task": "api.tasks.index.cleanup_indexes_task",
            "schedule": 24 * 60 * 60 * 7,  # Weekly
        },
    },
)

celery_app.autodiscover_tasks(["api.tasks"])
