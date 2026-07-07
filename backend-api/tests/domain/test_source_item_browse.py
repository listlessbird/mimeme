from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from tests.factories import create_ingestion_source

from domain.source_item_browse import SourceItemBrowser

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


def test_list_items_empty_for_source(db_session: Session, mock_storage: MagicMock) -> None:
    source = create_ingestion_source(session=db_session)
    db_session.flush()

    page = SourceItemBrowser(mock_storage).list_items(source.id, limit=20, offset=0)

    assert page.items == []
    assert page.total == 0
    assert page.limit == 20
    assert page.offset == 0
