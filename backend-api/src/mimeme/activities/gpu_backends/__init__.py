from __future__ import annotations

import threading

from mimeme.activities.gpu_backends.protocol import GpuBackend

_instance: GpuBackend | None = None
_lock = threading.Lock()


def get_gpu_backend() -> GpuBackend:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from mimeme.shared.config import settings

                backend_type = getattr(settings, "gpu_backend", "local")

                if backend_type == "modal":
                    from mimeme.activities.gpu_backends.modal import ModalGpuBackend

                    _instance = ModalGpuBackend()
                else:
                    from mimeme.activities.gpu_backends.local import LocalGpuBackend

                    _instance = LocalGpuBackend()
    assert _instance is not None
    return _instance


__all__ = ["get_gpu_backend"]
