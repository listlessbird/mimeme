from __future__ import annotations

from mimeme.domain.adapters.base import Adapter
from mimeme.domain.adapters.meme_api import MemeApiAdapter


class UnknownAdapterKeyError(Exception):
    """Raised when a Source names an ``adapter_key`` with no registered Adapter."""


# The single source of truth for which Adapters exist. Adding a provider is a new
# pure Adapter class plus one entry here (ADR 0001) — no other wiring.
ADAPTERS: dict[str, Adapter] = {adapter.key: adapter for adapter in (MemeApiAdapter(),)}

KNOWN_ADAPTER_KEYS = frozenset(ADAPTERS)


def get_adapter(key: str) -> Adapter:
    try:
        return ADAPTERS[key]
    except KeyError:
        raise UnknownAdapterKeyError(key) from None
