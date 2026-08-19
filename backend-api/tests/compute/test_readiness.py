from __future__ import annotations

from mimeme.compute.app import create_app
from mimeme.config import Settings


def test_inference_role_readiness_route_is_exposed() -> None:
    app = create_app(Settings())

    assert "/v1/roles/inference/ready" in {route.path for route in app.routes}
