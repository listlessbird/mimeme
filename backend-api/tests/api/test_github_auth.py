from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from mimeme.api.github_oauth import GitHubIdentity, GitHubOAuthFailure
from mimeme.api.routers import auth
from mimeme.config import AuthConfig, Settings


@dataclass
class FakeGitHubOAuth:
    identity: GitHubIdentity | None = None
    failure: bool = False

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse(f"https://github.test/authorize?redirect_uri={redirect_uri}")

    async def fetch_identity(self, request: Request) -> GitHubIdentity:
        if self.failure or self.identity is None:
            raise GitHubOAuthFailure("fake OAuth failure")
        return self.identity


def _client(
    *,
    allowed_ids: frozenset[str] = frozenset({"12345"}),
    oauth: FakeGitHubOAuth | None = None,
    app_env: str = "test",
) -> tuple[TestClient, Settings]:
    settings = Settings(
        app_env=app_env,
        auth=AuthConfig(
            allowed_github_ids=allowed_ids,
            github_callback_url="https://api.test/auth/github/callback",
            ui_url="https://ui.test",
        ),
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.github_oauth = oauth
    app.include_router(auth.router)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret-with-enough-entropy",
        session_cookie=settings.auth.session_cookie,
        max_age=settings.auth.session_max_age_s,
        same_site="lax",
    )
    return TestClient(app, follow_redirects=False), settings


def test_github_identity_parses_provider_numeric_id() -> None:
    identity = GitHubIdentity.model_validate(
        {"id": 124798751, "login": "listlessbird", "avatar_url": None, "extra": "ignored"}
    )

    assert identity.id == "124798751"


def test_session_is_closed_without_cookie() -> None:
    client, _settings = _client()

    response = client.get("/auth/session")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False, "dev_open": False, "user": None}


def test_development_session_is_open_without_oauth_configuration() -> None:
    client, _settings = _client(app_env="development")

    response = client.get("/auth/session")

    assert response.json() == {"authenticated": True, "dev_open": True, "user": None}


def test_callback_creates_session_for_allowlisted_github_id() -> None:
    identity = GitHubIdentity(id="12345", login="ril", avatar_url="https://avatar.test/ril")
    client, _settings = _client(oauth=FakeGitHubOAuth(identity=identity))

    callback = client.get("/auth/github/callback")
    session = client.get("/auth/session")

    assert callback.status_code == 303
    assert callback.headers["location"] == "https://ui.test/admin/sources"
    assert "mimeme_admin_session=" in callback.headers["set-cookie"]
    assert session.json() == {
        "authenticated": True,
        "dev_open": False,
        "user": {
            "id": "12345",
            "login": "ril",
            "avatar_url": "https://avatar.test/ril",
        },
    }


def test_callback_rejects_github_id_outside_allowlist() -> None:
    identity = GitHubIdentity(id="67890", login="other", avatar_url=None)
    client, _settings = _client(oauth=FakeGitHubOAuth(identity=identity))

    callback = client.get("/auth/github/callback")
    session = client.get("/auth/session")

    assert callback.status_code == 303
    assert callback.headers["location"] == "https://ui.test/admin-unlock?error=denied"
    assert session.json()["authenticated"] is False


def test_allowlist_change_revokes_existing_session() -> None:
    identity = GitHubIdentity(id="12345", login="ril", avatar_url=None)
    client, settings = _client(oauth=FakeGitHubOAuth(identity=identity))
    client.get("/auth/github/callback")

    settings.auth.allowed_github_ids = frozenset()

    assert client.get("/auth/session").json()["authenticated"] is False


def test_oauth_failure_returns_recoverable_login_error() -> None:
    client, _settings = _client(oauth=FakeGitHubOAuth(failure=True))

    response = client.get("/auth/github/callback")

    assert response.status_code == 303
    assert response.headers["location"] == "https://ui.test/admin-unlock?error=oauth"


def test_logout_clears_session() -> None:
    identity = GitHubIdentity(id="12345", login="ril", avatar_url=None)
    client, _settings = _client(oauth=FakeGitHubOAuth(identity=identity))
    client.get("/auth/github/callback")

    logout = client.post("/auth/logout")

    assert logout.status_code == 303
    assert logout.headers["location"] == "https://ui.test/admin-unlock"
    assert client.get("/auth/session").json()["authenticated"] is False
