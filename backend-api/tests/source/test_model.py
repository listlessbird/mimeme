from __future__ import annotations

from mimeme.db.schema import DuplicateReason, ProcessingStatus, SourceRunStatus
from mimeme.source.model import (
    DiscoveredItem,
    UrlOutcome,
    dedup_source_items,
    derive_run_accounting,
)


def _item(ext: str) -> DiscoveredItem:
    return DiscoveredItem(external_item_id=ext, media_url=f"https://a/{ext}.jpg")


class TestDedup:
    def test_splits_new_and_seen(self) -> None:
        result = dedup_source_items([_item("a"), _item("b")], seen_ids={"b"})
        assert [i.external_item_id for i in result.new] == ["a"]
        assert [i.external_item_id for i in result.already_seen] == ["b"]

    def test_collapses_intra_batch_duplicates(self) -> None:
        result = dedup_source_items([_item("a"), _item("a"), _item("b")], seen_ids=set())
        assert [i.external_item_id for i in result.new] == ["a", "b"]
        assert result.already_seen == []


class TestAccounting:
    def test_no_failures_is_completed(self) -> None:
        outcomes = [UrlOutcome(status=ProcessingStatus.DONE) for _ in range(3)]
        acc = derive_run_accounting(discovered_items=3, url_outcomes=outcomes)
        assert acc.status == SourceRunStatus.COMPLETED
        assert (acc.discovered, acc.queued, acc.failed) == (3, 3, 0)

    def test_empty_run_is_completed(self) -> None:
        acc = derive_run_accounting(discovered_items=0, url_outcomes=[])
        assert acc.status == SourceRunStatus.COMPLETED
        assert acc.queued == 0

    def test_all_failed_is_failed(self) -> None:
        outcomes = [UrlOutcome(status=ProcessingStatus.FAILED) for _ in range(2)]
        acc = derive_run_accounting(discovered_items=2, url_outcomes=outcomes)
        assert acc.status == SourceRunStatus.FAILED
        assert acc.failed == 2

    def test_some_failed_is_partial(self) -> None:
        outcomes = [
            UrlOutcome(status=ProcessingStatus.DONE),
            UrlOutcome(status=ProcessingStatus.FAILED),
        ]
        acc = derive_run_accounting(discovered_items=2, url_outcomes=outcomes)
        assert acc.status == SourceRunStatus.PARTIAL
        assert (acc.queued, acc.failed) == (2, 1)

    def test_counts_duplicates(self) -> None:
        outcomes = [
            UrlOutcome(status=ProcessingStatus.DONE, duplicate_reason=DuplicateReason.SHA256),
            UrlOutcome(status=ProcessingStatus.DONE),
        ]
        acc = derive_run_accounting(discovered_items=2, url_outcomes=outcomes)
        assert acc.duplicate == 1
