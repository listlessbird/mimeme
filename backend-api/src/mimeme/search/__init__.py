from __future__ import annotations

from typing import TYPE_CHECKING

from mimeme.search.activation import activate, reconcile
from mimeme.search.client import Activation, Client
from mimeme.search.error import (
    Error,
    Failed,
    Incompatible,
    Invalid,
    Loading,
    NotFound,
    Stale,
    Unavailable,
)
from mimeme.search.model import (
    Batch,
    Candidate,
    CandidateRequest,
    Encoder,
    File,
    Load,
    Loaded,
    Page,
    Query,
    Result,
    Rollback,
    Status,
    Switch,
)
from mimeme.search.run import Rows, run

if TYPE_CHECKING:
    import httpx

    from mimeme.shared.config import Settings


def create(settings: Settings, http: httpx.AsyncClient) -> Client:
    from mimeme.search.remote import Remote

    return Remote(http, base_url=settings.compute.gateway_url)


__all__ = [
    "Activation",
    "Batch",
    "Candidate",
    "CandidateRequest",
    "Client",
    "Encoder",
    "Error",
    "Failed",
    "File",
    "Load",
    "Loaded",
    "Incompatible",
    "Invalid",
    "Loading",
    "NotFound",
    "Page",
    "Query",
    "Result",
    "Rollback",
    "Rows",
    "Status",
    "Stale",
    "Switch",
    "Unavailable",
    "activate",
    "create",
    "reconcile",
    "run",
]
