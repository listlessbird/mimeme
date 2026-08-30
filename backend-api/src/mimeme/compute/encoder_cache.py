from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

import numpy as np

from mimeme.search.model import Encoder


class Session(Protocol):
    source_model: str

    def encode(self, text: str) -> np.ndarray: ...


Factory = Callable[[Encoder], Session]


@dataclass(frozen=True)
class Identity:
    repo: str
    revision: str
    variant: str
    threads: int


@dataclass
class _Entry:
    session: Session
    references: int


class Cache:
    def __init__(self, factory: Factory) -> None:
        self._factory = factory
        self._entries: dict[Identity, _Entry] = {}
        self._lock = RLock()


class Lease:
    def __init__(self, cache: Cache, identity: Identity) -> None:
        self._cache = cache
        self._identity = identity
        self._released = False


def create(factory: Factory) -> Cache:
    return Cache(factory)


def identify(config: Encoder) -> Identity:
    return Identity(
        repo=config.repo,
        revision=config.revision,
        variant=config.variant,
        threads=config.threads,
    )


def acquire(cache: Cache, config: Encoder) -> Lease:
    identity = identify(config)
    with cache._lock:
        entry = cache._entries.get(identity)
        if entry is None:
            entry = _Entry(session=cache._factory(config), references=0)
            cache._entries[identity] = entry
        entry.references += 1
    return Lease(cache, identity)


def release(lease: Lease) -> None:
    cache = lease._cache
    with cache._lock:
        if lease._released:
            return
        lease._released = True
        entry = cache._entries.get(lease._identity)
        if entry is None:
            return
        entry.references -= 1
        if entry.references == 0:
            del cache._entries[lease._identity]


def identity(lease: Lease) -> Identity:
    return lease._identity


def source_model(lease: Lease) -> str:
    return _session(lease).source_model


def encode(lease: Lease, text: str) -> np.ndarray:
    return _session(lease).encode(text)


def _session(lease: Lease) -> Session:
    cache = lease._cache
    with cache._lock:
        if lease._released:
            raise RuntimeError("encoder lease has been released")
        entry = cache._entries.get(lease._identity)
        if entry is None:
            raise RuntimeError("encoder session is unavailable")
        return entry.session
