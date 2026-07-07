from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from domain.ingestion_browse import IngestionBrowser, IngestionView

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


def test_list_attempts_empty(mock_storage: MagicMock) -> None:
    page = IngestionBrowser(mock_storage).list_attempts(
        limit=20,
        offset=0,
        view=IngestionView.ALL,
    )

    assert page.rows == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0
