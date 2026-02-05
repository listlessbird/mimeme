from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from activities import CPU_ACTIVITIES
from shared.config import settings
from workflows import ALL_WORKFLOWS


async def main() -> None:
    client = await Client.connect(settings.temporal_host)

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue_cpu,
        workflows=ALL_WORKFLOWS,
        activities=CPU_ACTIVITIES,
    )
    print(f"Starting CPU worker on queue: {settings.temporal_task_queue_cpu}")

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
