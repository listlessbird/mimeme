from __future__ import annotations

from urllib.parse import quote


class MediaUrlResolver:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def resolve(self, key: str) -> str:
        normalized_key = key.lstrip("/")
        if not normalized_key:
            raise ValueError("media key must not be empty")
        return f"{self._base_url}/{quote(normalized_key, safe='/')}"
