from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageRecord:
    sha256: str
    rel_path: Path
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    phash: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "rel_path": str(self.rel_path),
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "phash": self.phash,
        }
