from __future__ import annotations

from typing import Protocol, cast

import httpx
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from starlette.requests import Request
from starlette.responses import Response

from mimeme.config import AuthConfig


class GitHubIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    login: str
    avatar_url: str | None

    @field_validator("id", mode="before")
    @classmethod
    def stringify_numeric_id(cls, value: object) -> object:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


class GitHubOAuthFailure(Exception):
    pass


class GitHubOAuthClient(Protocol):
    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response: ...

    async def fetch_identity(self, request: Request) -> GitHubIdentity: ...


class _RemoteApp(Protocol):
    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response: ...

    async def authorize_access_token(self, request: Request) -> dict[str, object]: ...

    async def get(
        self,
        url: str,
        *,
        token: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response: ...


class AuthlibGitHubOAuth:
    def __init__(self, config: AuthConfig) -> None:
        client_secret = config.github_client_secret
        if not config.github_client_id or client_secret is None:
            raise ValueError("GitHub OAuth credentials are not configured")

        registry = OAuth()
        registry.register(
            name="github",
            client_id=config.github_client_id,
            client_secret=client_secret.get_secret_value(),
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"code_challenge_method": "S256"},
        )
        remote = registry.create_client("github")
        if remote is None:
            raise RuntimeError("GitHub OAuth client registration failed")
        self._remote = cast(_RemoteApp, remote)

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        try:
            return await self._remote.authorize_redirect(request, redirect_uri)
        except OAuthError as exc:
            raise GitHubOAuthFailure("GitHub authorization could not be started") from exc

    async def fetch_identity(self, request: Request) -> GitHubIdentity:
        try:
            token = await self._remote.authorize_access_token(request)
            response = await self._remote.get(
                "user",
                token=token,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "mimeme",
                },
            )
            response.raise_for_status()
            identity = GitHubIdentity.model_validate(response.json())
        except (OAuthError, httpx.HTTPError, ValidationError, ValueError) as exc:
            raise GitHubOAuthFailure("GitHub identity verification failed") from exc

        return identity
