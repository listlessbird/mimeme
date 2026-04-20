from __future__ import annotations

from activities.sources.protocol import SourceAdapter
from activities.sources.providers.api_league import ApiLeagueAdapter
from activities.sources.providers.meme_api import MemeApiAdaper

_ADAPTERS: dict[str, SourceAdapter] = {
    "meme_api": MemeApiAdaper(),
    "api_league": ApiLeagueAdapter(),
}


def get_adapter(key: str) -> SourceAdapter:
    adapter = _ADAPTERS.get(key)

    if adapter is None:
        raise ValueError(f"Unknown adapter: {key!r}. Available: {sorted(_ADAPTERS.keys())}")

    return adapter
