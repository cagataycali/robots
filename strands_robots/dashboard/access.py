"""Who may call a dashboard route.

One dependency, ``require_session``, guards every route that is not the login
screen. Three ways in, in this order, and the first that answers wins:

1. A valid passkey session token (``auth.verify_token``), presented as the
   ``strands_dash`` cookie the login screen sets or, when there is no cookie,
   as ``Authorization: Bearer``. The cookie wins when both are present: a
   bearer is something a page script can hold, the ``HttpOnly`` cookie is not,
   so a script-held token never shadows the session the browser keeps. A
   handoff session (``via="handoff"``) is honoured from the cookie only.
   Query-string tokens are not read: they land in access logs.
2. The static ``security.auth_token`` from settings, compared in constant time.
3. The bootstrap proof - but only while ``auth.auth_enabled()`` is False (no
   passkey enrolled, no env override) and no static token is configured. The
   bearer must equal the first-enrollment proof (``auth._first_enrollment_proof``:
   the configured ``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`` or the ``0600`` file the
   server minted beside the credential store), AND the request must be *this
   machine's browser at this machine*: socket peer loopback, no proxy header, a
   loopback-shaped ``Host``, and an ``Origin`` (when a browser sends one) that
   names that same host. That is the fresh-install posture: the dashboard is
   usable on the machine it runs on by whoever can read that file, until the
   owner enrols a passkey, after which the store turns auth on and this branch
   closes on its own.

The proof is required because a loopback peer is not presence at the machine: a
same-host L4 forwarder (``socat``, ``ssh -L``, ``docker -p``, a DNAT rule) hands
every remote client a ``127.0.0.1`` peer and adds no header, and a second local
account has the socket too. Reading a ``0600`` file as the service user is the
act neither can perform (f002, the same reasoning that already guards the first
enrollment). The topology conditions stay because the most common loopback
caller that is not the operator is the operator's own browser running someone
else's page. A DNS-rebound name still arrives on the loopback socket, but its
``Host`` is the attacker's name; a cross-site ``fetch`` or ``WebSocket`` still
arrives on the loopback socket, but its ``Origin`` is the attacker's origin.
Neither can be forged by a page, so both are checked before the open posture
admits anyone.

Everything else is 401. There is no allow-list of paths inside this module: a
route that wants to be public does not take the dependency, so the set of open
routes is visible at the route table, not buried in a string list here.

WebSockets have one more rule, and it comes BEFORE any credential: the
handshake's ``Origin`` must name this host (:func:`socket_origin_is_self`), or,
when there is no ``Origin`` at all, the handshake must carry an explicit bearer.
WebSockets are exempt from CORS, the cross-origin write middleware never sees a
``websocket`` scope, and ``SameSite=Strict`` is scoped to the site, which ignores
the port, so a page on ``http://localhost:3000`` opens a socket here with the
operator's cookie attached. The ``Origin`` check used to live only inside the
open posture, so that cookie was admitted with the header never read (f022).
Every socket route calls :func:`admit_socket`, which owns both rules.
"""

from __future__ import annotations

import hmac
import time
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, WebSocket

from strands_robots.dashboard import auth, settings

COOKIE = "strands_dash"

_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")

#: The names a browser at this machine can have typed to reach a loopback bind.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Methods that change state; a cross-origin request with one of these is refused.
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _presented(request: Request) -> tuple[str, str]:
    """``(token, where)``: the cookie when there is one, else the bearer header; ``("", "")`` for neither."""
    cookie = request.cookies.get(COOKIE, "").strip()
    if cookie:
        return cookie, "cookie"
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and header[7:].strip():
        return header[7:].strip(), "bearer"
    return "", ""


def presented_token(request: Request) -> str:
    """The session token a request carries, or an empty string.

    The ``HttpOnly`` cookie first; the ``Authorization: Bearer`` header only
    when there is no cookie. Never the query string.
    """
    return _presented(request)[0]


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


def _hostname(authority: str) -> str:
    """``host[:port]`` -> ``host``, lower-cased; a bracketed IPv6 literal keeps its brackets."""
    authority = authority.strip().lower()
    if authority.startswith("["):
        return authority.split("]", 1)[0] + "]"
    if authority.count(":") == 1:
        return authority.rsplit(":", 1)[0]
    return authority


def host_is_loopback(request: Request) -> bool:
    """Whether the ``Host`` header names this machine.

    The socket peer says where the packets came from; ``Host`` says what name
    the browser resolved to get here. A DNS-rebinding page has the first right
    and the second wrong, so this is the check that stops it. Starlette's
    TestClient sends ``testserver`` from its ``testclient`` peer; that pair is
    the one non-loopback name admitted, and only from that peer.
    """
    host = _hostname(request.headers.get("host", ""))
    if host in _LOOPBACK_HOSTS:
        return True
    client = request.client.host if request.client else None
    return host == "testserver" and client == "testclient"


def origin_is_self(request: Request) -> bool:
    """Whether the ``Origin`` header, if a browser sent one, names this very host.

    Scripts and curl send no ``Origin`` and pass; a browser cannot omit or forge
    it, so a page from anywhere else fails here whatever the socket says.
    ``null`` (sandboxed frames, file:// pages) is another origin.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    host = request.headers.get("host", "")
    if not host:
        return False
    split = urlsplit(origin.strip().lower())
    return split.scheme in ("http", "https") and split.netloc == host.strip().lower()


def session_claims(request: Request) -> dict[str, Any] | None:
    """Claims of a valid presented session, or None. Never raises.

    A handoff session presented as a bearer is refused: the redeeming device
    holds it in its ``HttpOnly`` cookie, so a copy in a header was lifted
    from somewhere it was never meant to be.
    """
    token, where = _presented(request)
    if not token:
        return None
    try:
        claims = auth.verify_token(token)
    except HTTPException:
        return None
    if where != "cookie" and claims.get("via") == "handoff":
        return None
    return claims


def static_token_matches(request: Request) -> bool:
    """Whether the presented token equals ``security.auth_token`` (constant time)."""
    configured = settings.get("security", "auth_token")
    if not configured:
        return False
    return hmac.compare_digest(presented_token(request), str(configured))


def bootstrap_token_matches(request: Request) -> bool:
    """Whether the presented token is the first-enrollment proof (constant time).

    The expectation is the one ``auth.begin_registration`` checks: the
    configured ``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN``, else the ``0600`` file
    beside the credential store. A request that presents nothing is refused
    before the file is consulted, so an anonymous probe never mints it.
    """
    presented = presented_token(request)
    if not presented:
        return False
    _source, expected = auth._first_enrollment_proof()
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def open_posture(request: Request) -> bool:
    """Fresh install: no auth configured anywhere, the caller holds the bootstrap proof, and is this machine's own browser, at this machine.

    Every conjunct is necessary. The proof is what a forwarded or second-account
    caller cannot present; the topology checks are what a page from elsewhere
    in the operator's own browser cannot satisfy. Anything short of all of them
    is 401, never a narrower admission.
    """
    if auth.auth_enabled():
        return False
    if settings.get("security", "auth_token"):
        return False
    return (
        peer_is_loopback(request)
        and not came_through_a_proxy(request)
        and host_is_loopback(request)
        and origin_is_self(request)
        and bootstrap_token_matches(request)
    )


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


def socket_origin_is_self(ws: WebSocket) -> bool:
    """Whether a WebSocket handshake comes from this dashboard's own page, or from a script that says who it is.

    A browser always sends ``Origin`` on a handshake, so one that is present must
    name this host exactly as :func:`origin_is_self` demands (``null`` and a
    sibling port are other origins). One that is absent is not a browser page:
    it is admitted only when the handshake carries an explicit ``Authorization:
    Bearer``, never on the cookie a browser would have attached, so a replayed
    cookie with the header stripped is refused too.
    """
    if ws.headers.get("origin") is not None:
        return origin_is_self(ws)  # type: ignore[arg-type]  # WebSocket answers headers like a Request
    header = ws.headers.get("authorization", "")
    return header.lower().startswith("bearer ") and bool(header[7:].strip())


async def admit_socket(ws: WebSocket) -> dict[str, Any] | None:
    """The caller of a WebSocket handshake, or None once the handshake has been refused.

    Origin first, credential second, and the first refusal is final: a foreign
    or missing ``Origin`` is a handshake rejection (never accepted, HTTP 403 on
    the wire) before any cookie or token is read, so the credential can neither
    rescue it nor be confirmed by it. A same-origin page without a session is
    then refused the way :func:`refuse_socket` documents, with 4401 after
    ``accept`` so the page can show the login screen.
    """
    if not socket_origin_is_self(ws):
        tally = getattr(ws.app.state, "refusals", None)
        if tally is not None:
            tally.record(client=(ws.client.host if ws.client else "?"), path=ws.url.path, now=time.time())
        await ws.close(code=4403)
        return None
    try:
        who = caller(ws)  # type: ignore[arg-type]  # WebSocket answers headers like a Request
    except HTTPException:
        await refuse_socket(ws, 4401)
        return None
    return who


async def refuse_socket(ws: WebSocket, code: int) -> None:
    """Refuse *ws* so that the page which opened it can read why.

    A close sent before ``accept`` is a handshake rejection, not a close: the
    ASGI server answers the handshake with HTTP 403 and no close code ever
    reaches the wire, so a browser reports 1006 for "sign in required" and "no
    such session" alike. The page needs that difference - ``static/app.js``
    shows the login screen on 4401 and nothing on the rest - so the socket is
    accepted and then closed with *code*. No frame is ever sent on it.

    A caller from another origin is the exception, and is refused at the
    handshake: WebSockets are exempt from CORS, so a page anywhere can open one
    here, and it is owed neither an accepted socket nor the reason.

    Args:
        ws: The socket to refuse. It must not have been accepted yet.
        code: The application close code, 4000-4999.
    """
    if code == 4401:
        tally = getattr(ws.app.state, "refusals", None)
        if tally is not None:
            tally.record(client=(ws.client.host if ws.client else "?"), path=ws.url.path, now=time.time())
    if origin_is_self(ws):  # type: ignore[arg-type]  # WebSocket answers headers like a Request
        await ws.accept()
    await ws.close(code=code)


async def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency: the caller, or 401."""
    return caller(request)
