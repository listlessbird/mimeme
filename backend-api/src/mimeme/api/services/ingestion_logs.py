from __future__ import annotations

import datetime
from typing import Any

import httpx
import structlog

from mimeme.config import Settings

log = structlog.get_logger()

_KNOWN_KEYS = (
    "event",
    "level",
    "activity_name",
    "workflow_name",
    "step",
    "last_step",
    "outcome",
    "duration_ms",
    "error",
    "attempt",
    "workflow_id",
    "run_id",
    "image_id",
)


class IngestionLogEntry:
    __slots__ = (
        "time",
        "level",
        "event",
        "activity_name",
        "step",
        "outcome",
        "duration_ms",
        "error",
        "attempt",
        "workflow_id",
        "data",
    )

    def __init__(self, time: str, data: dict[str, Any]) -> None:
        self.time = time
        self.level = _str(data.get("level"))
        self.event = _str(data.get("event"))
        self.activity_name = _str(data.get("activity_name"))
        self.step = _str(data.get("step") or data.get("last_step") or data.get("activity_name"))
        self.outcome = _str(data.get("outcome"))
        duration_ms = data.get("duration_ms")
        self.duration_ms = (
            round(float(duration_ms))
            if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool)
            else None
        )
        self.error = _str(data.get("error"))
        self.attempt = data.get("attempt") if isinstance(data.get("attempt"), int) else None
        self.workflow_id = _str(data.get("workflow_id"))
        self.data = data


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class AxiomLogReader:
    def __init__(self, settings: Settings, http: httpx.AsyncClient) -> None:
        self._token = settings.logging.axiom_query_token.get_secret_value()
        self._dataset = settings.logging.axiom_dataset
        self._http = http

    @property
    def available(self) -> bool:
        return bool(self._token and self._dataset)

    async def fetch_attempt_logs(
        self,
        *,
        ingest_url_id: int,
        job_id: str,
        created_at: datetime.datetime | None,
        limit: int,
    ) -> list[IngestionLogEntry]:
        if not self.available:
            return []

        safe_job_id = job_id.replace("'", "")
        apl = (
            f"['{self._dataset}'] "
            f"| where ingest_url_id == {int(ingest_url_id)} or job_id == '{safe_job_id}' "
            f"| sort by _time asc "
            f"| limit {int(limit)}"
        )

        payload: dict[str, object] = {"apl": apl}
        if created_at is not None:
            payload["startTime"] = (created_at - datetime.timedelta(hours=1)).isoformat()
            payload["endTime"] = (
                datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
            ).isoformat()

        try:
            response = await self._http.post(
                "https://api.axiom.co/v1/datasets/_apl",
                params={"format": "legacy"},
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            log.warning("ingestion_logs_query_failed", ingest_url_id=ingest_url_id, error=str(exc))
            return []

        entries: list[IngestionLogEntry] = []
        for match in result.get("matches", []):
            if not isinstance(match, dict):
                continue
            data = match.get("data")
            timestamp = match.get("_time")
            if isinstance(data, dict) and isinstance(timestamp, str):
                entries.append(IngestionLogEntry(time=timestamp, data=data))
        return entries


__all__ = ["AxiomLogReader", "IngestionLogEntry"]
