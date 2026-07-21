"""Tests for the Adapter registry (issue 03, slice 4).

The registry is the single source of truth for which Adapters exist — the real
replacement for the KNOWN_ADAPTERS frozenset placeholder. SourceRegistry.create
validates against it and discover_and_queue_activity resolves an Adapter from it.
"""

from __future__ import annotations

import pytest

from mimeme.domain.adapters.meme_api import MemeApiAdapter
from mimeme.domain.adapters.registry import (
    ADAPTERS,
    KNOWN_ADAPTER_KEYS,
    UnknownAdapterKeyError,
    get_adapter,
)


def test_get_adapter_returns_the_meme_api_adapter() -> None:
    assert isinstance(get_adapter("meme_api"), MemeApiAdapter)


def test_unknown_key_raises() -> None:
    with pytest.raises(UnknownAdapterKeyError):
        get_adapter("does_not_exist")


def test_known_adapter_keys_contains_meme_api() -> None:
    assert "meme_api" in KNOWN_ADAPTER_KEYS


def test_registry_is_keyed_by_each_adapter_own_key() -> None:
    for key, adapter in ADAPTERS.items():
        assert adapter.key == key
