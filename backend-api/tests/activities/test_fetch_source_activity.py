"""Tests for fetch_source_activity (issue 03, slice 3).

The generic Source fetch: execute one Adapter-described FetchRequest over HTTP
and hand back the raw JSON for the Adapter to parse. It knows nothing about
``meme_api`` — only how to perform the request and classify the failure.

Error semantics mirror download_image_activity (prior art in
test_storage_activities.py):
  - non-retryable 4xx / unusable payload  -> success=False (run continues)
  - 5xx, 429/408/425, transport errors     -> raise (Temporal retries)
"""

from __future__ import annotations

from http import HTTPMethod
from unittest.mock import MagicMock, patch

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from mimeme.activities.ingestion.activities import fetch_source_activity
from mimeme.activities.ingestion.models import FetchSourceInput
from mimeme.domain.adapters.base import FetchRequest


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


def _fake_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.request.return_value = response
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


def _input(url: str = "https://meme-api.com/gimme/memes/50") -> FetchSourceInput:
    return FetchSourceInput(request=FetchRequest(url=url), source_run_id=7)


class TestFetchSourceActivity:
    def test_successful_fetch_returns_raw_json(self, activity_env: ActivityEnvironment) -> None:
        payload = {"count": 1, "memes": [{"postLink": "https://redd.it/abc"}]}
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = payload

        with patch(
            "mimeme.activities.ingestion.activities.httpx.Client",
            return_value=_fake_client(response),
        ):
            result = activity_env.run(fetch_source_activity, _input())

        assert result.success is True
        assert result.status_code == 200
        assert result.raw == payload
        assert result.error is None

    def test_request_uses_method_and_url_from_fetch_request(
        self, activity_env: ActivityEnvironment
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {"memes": []}
        client = _fake_client(response)

        inp = FetchSourceInput(
            request=FetchRequest(url="https://meme-api.com/gimme/aww/10", method=HTTPMethod.GET)
        )

        with patch("mimeme.activities.ingestion.activities.httpx.Client", return_value=client):
            activity_env.run(fetch_source_activity, inp)

        method, url = client.request.call_args.args
        assert str(method) == "GET"
        assert url == "https://meme-api.com/gimme/aww/10"

    def test_terminal_4xx_returns_failure(self, activity_env: ActivityEnvironment) -> None:
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=response
        )

        with patch(
            "mimeme.activities.ingestion.activities.httpx.Client",
            return_value=_fake_client(response),
        ):
            result = activity_env.run(fetch_source_activity, _input())

        assert result.success is False
        assert result.status_code == 404
        assert result.raw is None
        assert result.error is not None

    def test_retryable_5xx_raises(self, activity_env: ActivityEnvironment) -> None:
        response = MagicMock()
        response.status_code = 500
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=response
        )

        with (
            patch(
                "mimeme.activities.ingestion.activities.httpx.Client",
                return_value=_fake_client(response),
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            activity_env.run(fetch_source_activity, _input())

    def test_rate_limit_429_is_retryable_and_raises(
        self, activity_env: ActivityEnvironment
    ) -> None:
        response = MagicMock()
        response.status_code = 429
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=response
        )

        with (
            patch(
                "mimeme.activities.ingestion.activities.httpx.Client",
                return_value=_fake_client(response),
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            activity_env.run(fetch_source_activity, _input())

    def test_transport_error_raises_for_retry(self, activity_env: ActivityEnvironment) -> None:
        client = MagicMock()
        client.request.side_effect = httpx.ConnectError("Connection refused")
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        with (
            patch("mimeme.activities.ingestion.activities.httpx.Client", return_value=client),
            pytest.raises(httpx.ConnectError),
        ):
            activity_env.run(fetch_source_activity, _input())

    def test_invalid_json_returns_failure(self, activity_env: ActivityEnvironment) -> None:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("Expecting value")

        with patch(
            "mimeme.activities.ingestion.activities.httpx.Client",
            return_value=_fake_client(response),
        ):
            result = activity_env.run(fetch_source_activity, _input())

        assert result.success is False
        assert result.error is not None

    def test_non_object_json_returns_failure(self, activity_env: ActivityEnvironment) -> None:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = ["not", "an", "object"]

        with patch(
            "mimeme.activities.ingestion.activities.httpx.Client",
            return_value=_fake_client(response),
        ):
            result = activity_env.run(fetch_source_activity, _input())

        assert result.success is False
        assert result.error is not None
