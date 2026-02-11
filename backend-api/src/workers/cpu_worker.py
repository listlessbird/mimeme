from __future__ import annotations

import asyncio

import structlog
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities import ALL_ACTIVITIES, CPU_ACTIVITIES, GPU_ACTIVITIES
from shared.config import settings
from shared.logging import setup_logging
from workflows import ALL_WORKFLOWS


async def main() -> None:
    setup_logging("worker-cpu")
    log = structlog.get_logger()

    use_modal = getattr(settings, "gpu_backend", "local") == "modal"

    client = await Client.connect(
        settings.temporal_host,
        data_converter=pydantic_data_converter,
    )

    if use_modal:
        log.info(
            "worker_started",
            worker_type="cpu+modal",
            cpu_queue=settings.temporal_task_queue_cpu,
            gpu_queue=settings.temporal_task_queue_gpu,
            activities=len(ALL_ACTIVITIES),
            workflows=len(ALL_WORKFLOWS),
        )

        cpu_worker = Worker(
            client,
            task_queue=settings.temporal_task_queue_cpu,
            workflows=ALL_WORKFLOWS,
            activities=ALL_ACTIVITIES,
        )

        if settings.temporal_task_queue_gpu != settings.temporal_task_queue_cpu:
            gpu_worker = Worker(
                client, task_queue=settings.temporal_task_queue_gpu, activities=GPU_ACTIVITIES
            )

            await asyncio.gather(cpu_worker.run(), gpu_worker.run())
        else:
            await cpu_worker.run()
    else:
        log.info(
            "worker_started",
            worker_type="cpu",
            task_queue=settings.temporal_task_queue_cpu,
            activities=len(CPU_ACTIVITIES),
            workflows=len(ALL_WORKFLOWS),
        )

        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue_cpu,
            workflows=ALL_WORKFLOWS,
            activities=CPU_ACTIVITIES,
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
