from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import time

import structlog
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from shared.config import settings
from shared.logging import setup_logging


def _preload_native_image_libs(log: structlog.BoundLogger) -> tuple[int, int]:
    loaded = 0
    failed = 0
    # Preload system libs to avoid vendored wheel ABI conflicts (libjpeg/libtiff/libvips).
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
    started = time.monotonic()
    outcome = "success"
    error_type: str | None = None
    error_message: str | None = None

    setup_logging("worker-gpu")
    log = structlog.get_logger().bind(worker_type="gpu", task_queue=settings.temporal_task_queue_gpu)
    log.info("worker_step", step="startup_begin")
    loaded_libs, failed_libs = _preload_native_image_libs(log)
    log.info("worker_step", step="native_preload_complete", loaded_libs=loaded_libs, failed_libs=failed_libs)

    # Import after native preload so downstream model imports resolve the same ABI set.
    from activities import GPU_ACTIVITIES

    log.info("worker_step", step="connect_temporal", temporal_host=settings.temporal_host)
    try:
        client = await Client.connect(
            settings.temporal_host,
            data_converter=pydantic_data_converter,
        )
        log.info("worker_step", step="temporal_connected")

        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue_gpu,
            activities=GPU_ACTIVITIES,
        )

        log.info(
            "worker_started",
            activities=len(GPU_ACTIVITIES),
        )
        log.info("worker_step", step="run_loop_start")
        await worker.run()
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        log.error(
            "worker_step",
            step="run_loop_failed",
            error_type=error_type,
            error=error_message,
            exc_info=True,
        )
        raise
    finally:
        log.info(
            "worker_wide_event",
            event_type="worker_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            loaded_libs=loaded_libs if "loaded_libs" in locals() else 0,
            failed_libs=failed_libs if "failed_libs" in locals() else 0,
            error_type=error_type,
            error=error_message,
        )


if __name__ == "__main__":
    asyncio.run(main())
