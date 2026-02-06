from __future__ import annotations

import asyncio

import structlog
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities import CPU_ACTIVITIES
from shared.config import settings
from shared.logging import setup_logging
from workflows import ALL_WORKFLOWS


async def main() -> None:
    setup_logging("worker-cpu")
    log = structlog.get_logger()

    client = await Client.connect(
        settings.temporal_host,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue_cpu,
        workflows=ALL_WORKFLOWS,
        activities=CPU_ACTIVITIES,
    )
    log.info(
        "worker_started",
        worker_type="cpu",
        task_queue=settings.temporal_task_queue_cpu,
        activities=len(CPU_ACTIVITIES),
        workflows=len(ALL_WORKFLOWS),
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
