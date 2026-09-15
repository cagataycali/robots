"""Who may call a dashboard route.

One dependency, ``require_session``, guards every route that is not the login
screen. Three ways in, in this order, and the first that answers wins:

1. A valid passkey session token (``auth.verify_token``), presented as
   ``Authorization: Bearer`` or as the ``strands_dash`` cookie the login screen
   sets. Query-string tokens are not read: they land in access logs.
2. The static ``security.auth_token`` from settings, compared in constant time.
3. Nothing at all - but only while ``auth.auth_enabled()`` is False (no passkey
   enrolled, no env override) AND the socket peer is loopback with no proxy
   header. That is the fresh-install posture: the dashboard is usable on the
   machine it runs on until the owner enrols a passkey, after which the store
   turns auth on and this branch closes on its own.

Everything else is 401. There is no allow-list of paths inside this module: a
route that wants to be public does not take the dependency, so the set of open
routes is visible at the route table, not buried in a string list here.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import HTTPException, Request

from strands_robots.dashboard import auth, settings

COOKIE = "strands_dash"

_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")


def presented_token(request: Request) -> str:
    """The session token a request carries, or an empty string.

    Bearer header first, then the cookie. Never the query string.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get(COOKIE, "").strip()


def came_through_a_proxy(request: Request) -> bool:
    """Whether a forwarding header says this connection was relayed."""
    return any(h in request.headers for h in _PROXY_HEADERS)


def peer_is_loopback(request: Request) -> bool:
    """Whether the socket peer is this machine, by address alone."""
    client = request.client.host if request.client else None
    if client is None:
        return False
    if client == "testclient":
        # Starlette's TestClient. Tests that want to see the closed posture set
        # STRANDS_DASH_AUTH_ENABLED=1 or enrol a credential; this is loopback.
        return True
    return bool(auth.client_is_loopback(client))


def session_claims(request: Request) -> dict[str, Any] | None:
    """Claims of a valid presented session, or None. Never raises."""
    token = presented_token(request)
    if not token:
        return None
    try:
        return auth.verify_token(token)
    except HTTPException:
        return None


def static_token_matches(request: Request) -> bool:
    """Whether the presented token equals ``security.auth_token`` (constant time)."""
    configured = settings.get("security", "auth_token")
    if not configured:
        return False
    return hmac.compare_digest(presented_token(request), str(configured))


def open_posture(request: Request) -> bool:
    """Fresh install: no auth configured anywhere AND the caller is this machine."""
    if auth.auth_enabled():
        return False
    if settings.get("security", "auth_token"):
        return False
    return peer_is_loopback(request) and not came_through_a_proxy(request)


def caller(request: Request) -> dict[str, Any]:
    """Who this is, or an HTTP 401 the route returns as-is.

    Returns:
        ``{"via": "passkey", ...claims}``, ``{"via": "token"}`` or
        ``{"via": "loopback"}``.
    """
    claims = session_claims(request)
    if claims is not None:
        return {"via": "passkey", **claims}
    if static_token_matches(request):
        return {"via": "token"}
    if open_posture(request):
        return {"via": "loopback"}
    raise HTTPException(401, "sign in required")


async def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency: the caller, or 401."""
    return caller(request)
