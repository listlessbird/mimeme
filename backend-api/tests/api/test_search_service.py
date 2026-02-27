from __future__ import annotations

from api.services import search as search_module
from api.services.search import SearchService


class _NoTextIndexManager:
    is_loaded = True
    active_version = "v-test-index"

    def search(self, _query_vector, k: int = 20) -> list[tuple[int, float]]:
        return []

    def has_text_index(self) -> bool:
        return False

    def search_text(self, _query_vector, k: int = 20) -> list[tuple[int, float]]:
        raise AssertionError("search_text should not be called when text index is missing")


def test_hybrid_logs_warning_when_text_index_missing(monkeypatch) -> None:
    warnings: list[tuple[str, dict]] = []

    class _LogCapture:
        def warning(self, event: str, **kwargs) -> None:
            warnings.append((event, kwargs))

    monkeypatch.setattr(search_module, "log", _LogCapture())

    service = SearchService(_NoTextIndexManager())
    results = service.search_by_embedding(
        embedding=[0.1, 0.2, 0.3],
        db=None,  # type: ignore[arg-type]
        mode="hybrid",
    )

    assert results == []
    assert len(warnings) == 1
    event, fields = warnings[0]
    assert event == "search_hybrid_text_index_unavailable"
    assert fields["mode"] == "hybrid"
    assert fields["index_version"] == "v-test-index"
