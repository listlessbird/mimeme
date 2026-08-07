from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict

from mimeme.compute.model import ImageInfo, StorageRole
from mimeme.ingest.model import InvalidImage, Retryable


class Facts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    phash: str
    width: int
    height: int
    format: str
    mode: str

    @classmethod
    def from_info(cls, info: ImageInfo) -> Facts:
        return cls(
            sha256=info.sha256,
            phash=info.phash,
            width=info.width,
            height=info.height,
            format=info.format,
            mode=info.mode,
        )


@runtime_checkable
class Images(Protocol):
    async def inspect(self, key: str, *, role: StorageRole = "artifacts") -> Facts: ...


class ComputeImages:
    def __init__(self, http: httpx.AsyncClient, *, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def inspect(self, key: str, *, role: StorageRole = "artifacts") -> Facts:
        try:
            resp = await self._http.post(
                f"{self._base}/v1/image/inspect", json={"key": key, "role": role}
            )
        except httpx.TimeoutException as exc:
            raise Retryable(f"compute inspect timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise Retryable(f"compute inspect unavailable: {exc}") from exc
        if resp.status_code >= 500:
            raise Retryable(f"compute inspect {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise InvalidImage(f"compute inspect {resp.status_code}: {resp.text}")
        try:
            return Facts.from_info(ImageInfo.model_validate(resp.json()))
        except Exception as exc:
            raise Retryable(f"malformed compute inspect response: {exc}") from exc
