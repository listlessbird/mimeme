from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, base: Path, name: str) -> Workspace:
        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=base))
        return cls(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_atomic(self, name: str, data: bytes) -> Path:
        target = self.root / name
        fd, tmp_name = tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return target

    def read(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
