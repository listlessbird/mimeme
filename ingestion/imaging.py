from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

def image_info(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            w, h = img.size
            fmt = (img.format or "").lower() if img.format else None
            return w, h, fmt
    except Exception as e:
        print(f"Error getting image info for {path}: {e}")
        return None, None, None
    
