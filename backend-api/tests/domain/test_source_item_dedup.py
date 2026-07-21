"""Executable spec for Source-item dedup (issue 03, slice 2).

The first of the three ordered dedup layers (PRD §"Dedup — three ordered
layers"), and the only pre-download one. Pure: given the items a run discovered
plus the set of ``external_item_id``s this Source has already recorded, it
partitions the *distinct* discovered items into:

    SourceItemDedup.new          — not seen before; these spawn ingest_urls + Job
    SourceItemDedup.already_seen — seen on a prior run; recorded again, never ingested

Two reductions happen, in this order of intent:

    1. within-run collapse  — the provider (or overlapping subreddit fetches) can
       return the same post twice in one run; duplicates by ``external_item_id``
       collapse to a single item, **first occurrence wins**, order preserved.
    2. cross-run seen-drop   — any distinct item whose id is in ``seen_ids`` lands
       in ``already_seen`` instead of ``new``.

Every distinct discovered item surfaces in exactly one partition (so the caller
can upsert a ``source_item`` for all of ``new + already_seen`` — that is how a
future run remembers it — while queueing ingest only for ``new``). Identity is
``external_item_id`` and nothing else; media URLs are never a dedup anchor.

Prior art: tests/domain/test_source_run_accounting.py (pure frozen-result module).
"""

from __future__ import annotations

from mimeme.domain.adapters.base import DiscoveredItem
from mimeme.domain.source_item_dedup import SourceItemDedup, dedup_source_items


def _item(external_item_id: str, *, title: str | None = None) -> DiscoveredItem:
    return DiscoveredItem(
        external_item_id=external_item_id,
        media_url=f"https://i.redd.it/{external_item_id}.png",
        canonical_item_url=f"https://redd.it/{external_item_id}",
        title=title,
        raw_metadata={"postLink": f"https://redd.it/{external_item_id}"},
    )


def _ids(items: list[DiscoveredItem]) -> list[str]:
    return [item.external_item_id for item in items]


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


def test_empty_discovered_returns_empty_partitions() -> None:
    result = dedup_source_items([], seen_ids=set())

    assert result == SourceItemDedup(new=[], already_seen=[])


def test_empty_seen_set_with_items_makes_them_all_new() -> None:
    result = dedup_source_items([_item("a"), _item("b")], seen_ids=set())

    assert _ids(result.new) == ["a", "b"]
    assert result.already_seen == []


# ---------------------------------------------------------------------------
# Cross-run seen-drop
# ---------------------------------------------------------------------------


def test_all_seen_when_every_id_is_in_seen_set() -> None:
    result = dedup_source_items([_item("a"), _item("b")], seen_ids={"a", "b"})

    assert result.new == []
    assert _ids(result.already_seen) == ["a", "b"]


def test_mixed_partitions_new_and_already_seen_by_id() -> None:
    discovered = [_item("a"), _item("b"), _item("c")]

    result = dedup_source_items(discovered, seen_ids={"b"})

    assert _ids(result.new) == ["a", "c"]
    assert _ids(result.already_seen) == ["b"]


# ---------------------------------------------------------------------------
# Within-run collapse (first occurrence wins, order preserved)
# ---------------------------------------------------------------------------


def test_within_run_duplicates_collapse_to_one() -> None:
    discovered = [_item("a"), _item("a"), _item("b")]

    result = dedup_source_items(discovered, seen_ids=set())

    assert _ids(result.new) == ["a", "b"]
    assert result.already_seen == []


def test_collapse_keeps_the_first_occurrence() -> None:
    discovered = [_item("a", title="first"), _item("a", title="second")]

    result = dedup_source_items(discovered, seen_ids=set())

    assert [item.title for item in result.new] == ["first"]


def test_within_run_duplicate_of_a_seen_item_collapses_once_into_already_seen() -> None:
    discovered = [_item("a"), _item("a")]

    result = dedup_source_items(discovered, seen_ids={"a"})

    assert result.new == []
    assert _ids(result.already_seen) == ["a"]


# ---------------------------------------------------------------------------
# Order + partition invariant on a realistic mixed batch
# ---------------------------------------------------------------------------


def test_order_preserved_within_each_partition() -> None:
    discovered = [_item("c"), _item("a"), _item("b")]

    result = dedup_source_items(discovered, seen_ids={"a"})

    assert _ids(result.new) == ["c", "b"]
    assert _ids(result.already_seen) == ["a"]


def test_every_distinct_item_lands_in_exactly_one_partition() -> None:
    # new("x"), seen("y"), within-run dup of new("x"), new("z"), dup of seen("y")
    discovered = [
        _item("x"),
        _item("y"),
        _item("x"),
        _item("z"),
        _item("y"),
    ]

    result = dedup_source_items(discovered, seen_ids={"y"})

    assert _ids(result.new) == ["x", "z"]
    assert _ids(result.already_seen) == ["y"]

    distinct_discovered = {item.external_item_id for item in discovered}
    partitioned = _ids(result.new) + _ids(result.already_seen)
    assert sorted(partitioned) == sorted(distinct_discovered)
    assert len(partitioned) == len(set(partitioned))
