from __future__ import annotations

from urllib.parse import urlencode

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse, Response

from mimeme.api.github_oauth import GitHubOAuthClient, GitHubOAuthFailure
from mimeme.api.models.auth import AdminSessionResponse, AdminUserResponse
from mimeme.config import Settings

router = APIRouter(prefix="/auth", tags=["Auth"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _oauth_client(request: Request) -> GitHubOAuthClient:
    client: GitHubOAuthClient | None = request.app.state.github_oauth
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub sign-in is not configured",
        )
    return client


def _ui_redirect(settings: Settings, path: str, **query: str) -> str:
    url = f"{settings.auth.ui_url.rstrip('/')}{path}"
    return f"{url}?{urlencode(query)}" if query else url


@router.get("/github/login")
async def github_login(request: Request) -> Response:
    settings = _settings(request)
    if settings.app_env == "development":
        return RedirectResponse(_ui_redirect(settings, "/admin/sources"))

    client = _oauth_client(request)
    try:
        response = await client.authorize_redirect(request, settings.auth.github_callback_url)
    except GitHubOAuthFailure:
        structlog.get_logger().exception("github_oauth_start_failed")
        return RedirectResponse(
            _ui_redirect(settings, "/admin-unlock", error="oauth"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return response


@router.get("/github/callback")
async def github_callback(request: Request) -> RedirectResponse:
    settings = _settings(request)
    client = _oauth_client(request)

    try:
        identity = await client.fetch_identity(request)
    except GitHubOAuthFailure:
        request.session.clear()
        structlog.get_logger().exception("github_oauth_callback_failed")
        return RedirectResponse(
            _ui_redirect(settings, "/admin-unlock", error="oauth"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    if identity.id not in settings.auth.allowed_github_ids:
        request.session.clear()
        structlog.get_logger().warning(
            "github_admin_denied",
            github_id=identity.id,
            github_login=identity.login,
        )
        return RedirectResponse(
            _ui_redirect(settings, "/admin-unlock", error="denied"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    request.session.clear()
    request.session.update(
        {
            "github_id": identity.id,
            "github_login": identity.login,
            "github_avatar_url": identity.avatar_url,
        }
    )
    structlog.get_logger().info(
        "github_admin_authenticated",
        github_id=identity.id,
        github_login=identity.login,
    )
    return RedirectResponse(
        _ui_redirect(settings, "/admin/sources"),
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/session", response_model=AdminSessionResponse)
async def get_admin_session(request: Request) -> AdminSessionResponse:
    settings = _settings(request)
    if settings.app_env == "development":
        return AdminSessionResponse(authenticated=True, dev_open=True)

    github_id = request.session.get("github_id")
    if not isinstance(github_id, str) or github_id not in settings.auth.allowed_github_ids:
        return AdminSessionResponse(authenticated=False, dev_open=False)

    login = request.session.get("github_login")
    avatar_url = request.session.get("github_avatar_url")
    if not isinstance(login, str):
        request.session.clear()
        return AdminSessionResponse(authenticated=False, dev_open=False)

    return AdminSessionResponse(
        authenticated=True,
        dev_open=False,
        user=AdminUserResponse(
            id=github_id,
            login=login,
            avatar_url=avatar_url if isinstance(avatar_url, str) else None,
        ),
    )


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    settings = _settings(request)
    request.session.clear()
    return RedirectResponse(
        _ui_redirect(settings, "/admin-unlock"),
        status_code=status.HTTP_303_SEE_OTHER,
    )
