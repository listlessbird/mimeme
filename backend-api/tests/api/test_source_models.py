from __future__ import annotations

from datetime import UTC, datetime

from mimeme.api.models.sources import CreateSourceRequest, SourceResponse
from mimeme.api.routers.sources import _public_source_payload


def test_tumblr_source_config_is_accepted() -> None:
    request = CreateSourceRequest(
        name="tumblr memes",
        adapter_key="tumblr_tagged",
        adapter_config={
            "tags": ["meme", "reaction image"],
            "api_key": "tumblr-key",
            "min_note_count": 100,
        },
    )

    assert request.adapter_config["tags"] == ["meme", "reaction image"]


def test_tumblr_api_key_is_redacted_from_source_responses() -> None:
    payload = _public_source_payload(
        {
            "id": 1,
            "name": "tumblr memes",
            "adapter_key": "tumblr_tagged",
            "adapter_config": {"tags": ["meme"], "api_key": "tumblr-key"},
            "dataset": None,
            "schedule_cron": None,
            "schedule_timezone": "UTC",
            "max_items_per_run": 20,
            "enabled": True,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
    )

    response = SourceResponse.model_validate(payload)

    assert response.adapter_config["api_key"] == "[REDACTED]"
