from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog
from temporalio.worker import Worker

from mimeme.config import Settings
from mimeme.env import Env
from mimeme.index.activity import Activities as IndexActivities
from mimeme.index.schedule import Spec as IndexScheduleSpec
from mimeme.index.schedule import Temporal as IndexSchedule
from mimeme.index.workflow import RebuildWorkflow
from mimeme.ingest.activity import IngestActivities
from mimeme.ingest.workflow import IngestWorkflow
from mimeme.logging import setup_logging
from mimeme.source import schedule as source_schedule
from mimeme.source import store as source_store
from mimeme.source.activity import SourceActivities
from mimeme.source.workflow import SourceRetryWorkflow, SourceSyncWorkflow


def registrations(
    env: Any, *, poll_interval_s: float = 5.0
) -> tuple[list[type], list[Callable[..., Any]]]:
    ingest = IngestActivities(env, poll_interval_s=poll_interval_s)
    source = SourceActivities(env)
    index = IndexActivities(env)
    return (
        [IngestWorkflow, SourceSyncWorkflow, SourceRetryWorkflow, RebuildWorkflow],
        [
            ingest.item,
            ingest.finish,
            source.discover,
            source.finish,
            index.prepare,
            index.build,
            index.activate,
        ],
    )


async def main() -> None:
    settings = Settings()
    setup_logging(settings, "worker")
    log = structlog.get_logger()
    env = await Env.create(settings)

    workflows, activities = registrations(env, poll_interval_s=settings.compute.poll_interval_s)

    try:
        desired = await source_store.list_schedule_specs(env.db)
        await source_schedule.reconcile(
            source_schedule.TemporalScheduleStore(env.temporal), desired=desired
        )
        await IndexSchedule(env.temporal, task_queue=settings.temporal.task_queue).reconcile(
            IndexScheduleSpec(
                enabled=settings.index.rebuild_schedule_enabled,
                cron=settings.index.rebuild_schedule_cron,
                timezone=settings.index.rebuild_schedule_timezone,
            ),
            model=settings.inference.embed_model,
            index_type=settings.index.type,
        )
        log.info(
            "worker_started",
            task_queue=settings.temporal.task_queue,
            activities=len(activities),
            workflows=len(workflows),
        )
        await Worker(
            env.temporal,
            task_queue=settings.temporal.task_queue,
            workflows=workflows,
            activities=activities,
        ).run()
    finally:
        await env.aclose()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
