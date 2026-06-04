from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor

import structlog
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from shared.config import settings
from shared.logging import setup_logging


def _preload_native_image_libs(log: structlog.BoundLogger) -> tuple[int, int]:
    loaded = 0
    failed = 0
    for lib in ("jpeg", "tiff", "vips"):
        lib_path = ctypes.util.find_library(lib)
        if not lib_path:
            log.warning("native_lib_not_found", lib=lib)
            failed += 1
            continue
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            log.info("native_lib_preloaded", lib=lib, path=lib_path)
            loaded += 1
        except OSError as exc:
            log.error("native_lib_preload_failed", lib=lib, path=lib_path, error=str(exc))
            failed += 1
    return loaded, failed


async def main() -> None:
    setup_logging("worker")
    log = structlog.get_logger()

    if settings.gpu_backend == "local":
        _preload_native_image_libs(log)

    from activities.storage.phash_index import get_phash_index
    from shared.db import session_scope

    with session_scope() as session:
        get_phash_index().load_from_db(session)

    from activities import ACTIVITIES
    from workflows import ALL_WORKFLOWS

    client = await Client.connect(
        settings.temporal_host,
        data_converter=pydantic_data_converter,
    )
    activity_executor = ThreadPoolExecutor(max_workers=64, thread_name_prefix="activity")

    log.info(
        "worker_started",
        task_queue=settings.temporal_task_queue,
        gpu_backend=settings.gpu_backend,
        activities=len(ACTIVITIES),
        workflows=len(ALL_WORKFLOWS),
    )

    try:
        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=ALL_WORKFLOWS,
            activities=ACTIVITIES,
            activity_executor=activity_executor,
        )
        await worker.run()
    finally:
        activity_executor.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
