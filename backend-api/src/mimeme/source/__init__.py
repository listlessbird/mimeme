# ruff: noqa: F401

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from mimeme.source import rule

if TYPE_CHECKING:
    from mimeme.source.adapter import (
        KNOWN_ADAPTER_KEYS,
        Adapter,
        MemeApiAdapter,
        TumblrTaggedAdapter,
        UnknownAdapterKey,
        get_adapter,
    )  # noqa: F401
    from mimeme.source.flip import FlipAdapter, FlipConfig, ListingMode  # noqa: F401
    from mimeme.source.model import (
        DiscoveredItem,
        DiscoverInput,
        DiscoverResult,
        DuplicateSourceName,
        Error,
        FetchRequest,
        FinishInput,
        FinishResult,
        NothingToRetry,
        RawResponse,
        Retryable,
        RetryInput,
        RetryPlan,
        RetryResult,
        RunNotFound,
        SourceDetail,
        SourceItemIngestState,
        SourceItemNotFound,
        SourceListItem,
        SourceNotFound,
        SourceView,
        SyncInput,
        SyncResult,
    )  # noqa: F401

_EXPORTS = {
    "Adapter": ("mimeme.source.adapter", "Adapter"),
    "KNOWN_ADAPTER_KEYS": ("mimeme.source.adapter", "KNOWN_ADAPTER_KEYS"),
    "MemeApiAdapter": ("mimeme.source.adapter", "MemeApiAdapter"),
    "TumblrTaggedAdapter": ("mimeme.source.adapter", "TumblrTaggedAdapter"),
    "UnknownAdapterKey": ("mimeme.source.adapter", "UnknownAdapterKey"),
    "get_adapter": ("mimeme.source.adapter", "get_adapter"),
    "FlipAdapter": ("mimeme.source.flip", "FlipAdapter"),
    "FlipConfig": ("mimeme.source.flip", "FlipConfig"),
    "ListingMode": ("mimeme.source.flip", "ListingMode"),
    "DiscoveredItem": ("mimeme.source.model", "DiscoveredItem"),
    "DiscoverInput": ("mimeme.source.model", "DiscoverInput"),
    "DiscoverResult": ("mimeme.source.model", "DiscoverResult"),
    "DuplicateSourceName": ("mimeme.source.model", "DuplicateSourceName"),
    "Error": ("mimeme.source.model", "Error"),
    "FetchRequest": ("mimeme.source.model", "FetchRequest"),
    "FinishInput": ("mimeme.source.model", "FinishInput"),
    "FinishResult": ("mimeme.source.model", "FinishResult"),
    "NothingToRetry": ("mimeme.source.model", "NothingToRetry"),
    "RawResponse": ("mimeme.source.model", "RawResponse"),
    "Retryable": ("mimeme.source.model", "Retryable"),
    "RetryInput": ("mimeme.source.model", "RetryInput"),
    "RetryPlan": ("mimeme.source.model", "RetryPlan"),
    "RetryResult": ("mimeme.source.model", "RetryResult"),
    "RunNotFound": ("mimeme.source.model", "RunNotFound"),
    "SourceDetail": ("mimeme.source.model", "SourceDetail"),
    "SourceItemIngestState": ("mimeme.source.model", "SourceItemIngestState"),
    "SourceItemNotFound": ("mimeme.source.model", "SourceItemNotFound"),
    "SourceListItem": ("mimeme.source.model", "SourceListItem"),
    "SourceNotFound": ("mimeme.source.model", "SourceNotFound"),
    "SourceView": ("mimeme.source.model", "SourceView"),
    "SyncInput": ("mimeme.source.model", "SyncInput"),
    "SyncResult": ("mimeme.source.model", "SyncResult"),
}

__all__ = [
    "Adapter",
    "DiscoverInput",
    "DiscoverResult",
    "DiscoveredItem",
    "DuplicateSourceName",
    "Error",
    "FetchRequest",
    "FinishInput",
    "FinishResult",
    "FlipAdapter",
    "FlipConfig",
    "KNOWN_ADAPTER_KEYS",
    "ListingMode",
    "MemeApiAdapter",
    "NothingToRetry",
    "RawResponse",
    "RetryInput",
    "RetryPlan",
    "RetryResult",
    "Retryable",
    "RunNotFound",
    "SourceDetail",
    "SourceItemIngestState",
    "SourceItemNotFound",
    "SourceListItem",
    "SourceNotFound",
    "SourceView",
    "SyncInput",
    "SyncResult",
    "TumblrTaggedAdapter",
    "UnknownAdapterKey",
    "get_adapter",
    "rule",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
