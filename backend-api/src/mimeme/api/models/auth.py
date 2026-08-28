from __future__ import annotations

from pydantic import BaseModel


class AdminUserResponse(BaseModel):
    id: str
    login: str
    avatar_url: str | None = None


class AdminSessionResponse(BaseModel):
    authenticated: bool
    dev_open: bool
    user: AdminUserResponse | None = None
