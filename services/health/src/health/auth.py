"""Bearer token authentication for the dashboard API.

Reads SE_DASHBOARD_TOKEN from settings. If the token is empty, all
API requests are rejected with 401 (prevents accidental open access).
"""

from __future__ import annotations

from core.config import get_settings
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing token",
    headers={"WWW-Authenticate": "Bearer"},
)


async def require_token(request: Request) -> None:
    """FastAPI dependency — raises 401 if token is absent or wrong."""
    settings = get_settings()
    expected = settings.dashboard_token.get_secret_value()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard token not configured",
        )

    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None or credentials.credentials != expected:
        raise _UNAUTHORIZED
