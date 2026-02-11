from __future__ import annotations

import threading

from activities.gpu_backends.protocol import GpuBackend

_instance: GpuBackend | None = None
_lock = threading.Lock()


def get_gpu_backend() -> GpuBackend:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from shared.config import settings

                backend_type = getattr(settings, "gpu_backend", "local")

                if backend_type == "modal":
                    from activities.gpu_backends.modal import ModalGpuBackend

                    _instance = ModalGpuBackend()
                else:
                    from activities.gpu_backends.local import LocalGpuBackend

                    _instance = LocalGpuBackend()
    assert _instance is not None
    return _instance


__all__ = ["get_gpu_backend"]
