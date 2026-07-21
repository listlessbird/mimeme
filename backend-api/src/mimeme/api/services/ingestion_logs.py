from __future__ import annotations

import asyncio
import datetime
from typing import Any

import structlog
from axiom_py import Client
from axiom_py.client import AplOptions

from mimeme.shared.runtime import settings

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
        self.step = _str(data.get("step") or data.get("last_step"))
        self.outcome = _str(data.get("outcome"))
        self.duration_ms = (
            data.get("duration_ms") if isinstance(data.get("duration_ms"), int) else None
        )
        self.error = _str(data.get("error"))
        self.attempt = data.get("attempt") if isinstance(data.get("attempt"), int) else None
        self.workflow_id = _str(data.get("workflow_id"))
        self.data = data


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class AxiomLogReader:
    def __init__(self, token: str | None = None, dataset: str | None = None) -> None:
        self._token = (
            token if token is not None else settings.logging.axiom_api_token.get_secret_value()
        )
        self._dataset = dataset if dataset is not None else settings.logging.axiom_dataset
        self._client: Client | None = None

    @property
    def available(self) -> bool:
        return bool(self._token and self._dataset)

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(token=self._token)
        return self._client

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

        opts: AplOptions | None = None
        if created_at is not None:
            opts = AplOptions(
                start_time=created_at - datetime.timedelta(hours=1),
                end_time=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
                limit=limit,
            )

        try:
            result = await asyncio.to_thread(lambda: self._get_client().query(apl, opts))
        except Exception as exc:
            log.warning("ingestion_logs_query_failed", ingest_url_id=ingest_url_id, error=str(exc))
            return []

        entries: list[IngestionLogEntry] = []
        for match in result.matches or []:
            entries.append(IngestionLogEntry(time=match._time, data=dict(match.data)))
        return entries


__all__ = ["AxiomLogReader", "IngestionLogEntry"]
