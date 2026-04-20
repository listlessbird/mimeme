from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from activities.sources.models import FetchSourceItemsOutput


class SourceAdapter(Protocol):
    secret_ref_names: tuple[str, ...]

    def fetch_latest(
        self,
        adapter_cfg: dict[str, Any],
        max_items: int,
        secrets: Mapping[str, str],
    ) -> FetchSourceItemsOutput: ...
