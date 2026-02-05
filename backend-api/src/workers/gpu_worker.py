from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from activities import GPU_ACTIVITIES
from shared.config import settings


async def main() -> None:
    client = await Client.connect(settings.temporal_host)

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue_gpu,
        activities=GPU_ACTIVITIES,
    )

    print(f"Starting GPU worker on queue: {settings.temporal_task_queue_gpu}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
