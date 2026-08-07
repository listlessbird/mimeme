from __future__ import annotations

from mimeme.api.main import create_app
from mimeme.config import Settings


def test_app_builds_openapi_without_lifespan_side_effects() -> None:
    app = create_app(Settings())
    assert app.openapi()["openapi"].startswith("3.")
