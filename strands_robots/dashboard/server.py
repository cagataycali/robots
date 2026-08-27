"""FastAPI app for the Strands Robots Dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from strands_robots.dashboard import config_api, consent, deploy, lan_hint, settings
from strands_robots.dashboard.build_info import build_info
from strands_robots.dashboard.churn_guard import (
    ChurnGuard,
    effective_cap,
    viewer_identity,
)
from strands_robots.dashboard.device_manager import DeviceManager
from strands_robots.dashboard.mesh_bridge import MeshBridge, silent_arms, stop_outcome
from strands_robots.dashboard.refusals import RefusalTally
from strands_robots.dashboard.teleop_health import published_frames, teleop_health
from strands_robots.dashboard.ws_observability import (
    CloseLogThrottle,
    cap_note,
    close_line,
    close_verdict,
    fps_cap,
)

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

# : These two have to outlive individual SOCKETS to be able to count them - that is the whole
# : requirement, and app.state satisfies it.
_CAMERA_CLOSE_LOG = CloseLogThrottle()
_CAMERA_CHURN = ChurnGuard()

# : Reachable without a token: liveness (so a client can discover that auth is : required at
# all), the WebAuthn ceremony endpoints (you cannot log in from : behind a wall that requires
# being logged in), and the static shell (which : renders the login prompt).
# /api/auth/register/* gates ITSELF: once a : passkey exists, enrolling another requires a
# valid session.
PUBLIC_PATHS = {
    "/api/health",
    "/api/auth/status",
    "/api/auth/register/begin",
    "/api/auth/register/finish",
    "/api/auth/login/begin",
    "/api/auth/login/finish",
}

try:  # passkey auth is optional at import time (needs webauthn + PyJWT)
    from strands_robots.dashboard import auth as dash_auth
except Exception:  # pragma: no cover - deps missing in minimal installs
    dash_auth = None  # type: ignore[assignment]


class TokenAuthMiddleware:
    """Bearer-token gate for /api and /ws, as raw ASGI."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def _renewing(self, scope: dict[str, Any], send: Any, dash_auth: Any, presented: str) -> Any:
        if scope.get("type") != "http":
            return send
        try:
            fresh = dash_auth.renew_if_due(presented)
        except Exception as exc:  # pragma: no cover - defence in depth
            logger.warning("[auth] session renewal failed (%r)", exc)
            return send
        if not fresh:
            return send

        async def _send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                names = {k.lower() for k, _ in headers}
                headers.append((b"x-session-token", fresh.encode()))
                # Without this the browser can SEE nothing: a header the fetch layer
                # cannot read is a renewal that silently never happens.
                if b"access-control-expose-headers" not in names:
                    headers.append((b"access-control-expose-headers", b"X-Session-Token"))
                message = {**message, "headers": headers}
            await send(message)

        return _send

    @staticmethod
    def _note_refusal(scope: dict[str, Any], path: str, kind: str) -> None:
        try:
            state = getattr(scope.get("app"), "state", None)
            tally = getattr(state, "refusals", None) if state is not None else None
            if tally is None:
                tally = _REFUSALS
            client = scope.get("client")
            tally.record(
                client=str(client[0]) if client else "?",
                path=path,
                kind=kind,
                now=time.time(),
            )
        except Exception:  # pragma: no cover - bookkeeping must never break the guard
            pass

    @staticmethod
    def _client_is_local(scope: dict[str, Any]) -> bool:
        # A reverse proxy (cloudflared tunnel) connects FROM loopback on behalf of the whole internet.
        headers = {k.decode().lower() for k, _ in scope.get("headers") or []}
        if headers & {"cf-connecting-ip", "x-forwarded-for", "x-real-ip"}:
            return False
        client = scope.get("client")
        host = client[0] if client else None
        if host is None or host == "testclient":
            # in-process ASGI test clients have no real peer address
            return True
        if dash_auth is not None:
            return dash_auth.client_is_loopback(host)
        return host in ("127.0.0.1", "::1", "localhost")

    @staticmethod
    def _presented(scope: dict[str, Any]) -> str:
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return cast(str, auth[7:].strip())
        if headers.get("x-dashboard-token"):
            return cast(str, headers["x-dashboard-token"].strip())
        # Browsers cannot set headers on a WebSocket handshake, so the query
        # string is the only channel there.
        query = parse_qs(scope.get("query_string", b"").decode())
        return cast(str, (query.get("token") or [""])[0].strip())

    @staticmethod
    def _cross_origin_refused(scope: dict[str, Any]) -> bool:
        """True when a BROWSER cross-origin request must be refused."""
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        origin = headers.get("origin", "").strip()
        if not origin:
            return False
        host = headers.get("host", "").strip().lower()
        netloc = urlsplit(origin).netloc.strip().lower()
        if netloc and netloc == host:
            return False  # same-origin
        # A WILDCARD IS NOT A WRITE PERMIT.
        allowed = settings.get("security", "cors_origins", []) or []
        named = {str(a).rstrip("/") for a in allowed if str(a) != "*"}
        if origin.rstrip("/") in named:
            return False
        if scope["type"] == "websocket":
            return True
        return scope.get("method", "GET").upper() not in ("GET", "HEAD", "OPTIONS")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        guarded = path.startswith("/api") or path.startswith("/ws")
        if not guarded or path in PUBLIC_PATHS or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if self._cross_origin_refused(scope):
            self._note_refusal(scope, path, "origin")
            if scope["type"] == "websocket":
                await receive()  # consume websocket.connect before rejecting
                await send({"type": "websocket.close", "code": 1008})
                return
            response = JSONResponse(
                {"detail": "cross-origin request refused - add the origin to security.cors_origins"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        token = settings.get("security", "auth_token")
        passkeys_on = dash_auth is not None and dash_auth.auth_enabled()
        if not token and not passkeys_on:
            # nothing configured: open for loopback only
            if self._client_is_local(scope):
                await self.app(scope, receive, send)
                return
        else:
            presented = self._presented(scope)
            if token and hmac.compare_digest(presented, str(token)):
                await self.app(scope, receive, send)
                return
            if dash_auth is not None and dash_auth.session_is_valid(presented):
                await self.app(scope, receive, self._renewing(scope, send, dash_auth, presented))
                return
        self._note_refusal(scope, path, "credential")
        if scope["type"] == "websocket":
            await receive()  # consume websocket.connect before rejecting
            await send({"type": "websocket.close", "code": 1008})
            return
        response = JSONResponse({"detail": "unauthorized"}, status_code=401)
        await response(scope, receive, send)


# : The frame types /ws/chat implements.
_REFUSALS = RefusalTally()

_CHAT_FRAME_TYPES = frozenset({"chat", "ping", "interrupt_response"})

CHAT_MAX_FRAME_BYTES = 32 * 1024  # a generous chat turn; 2 MB frames ran real model turns


def parse_chat_frame(message: dict[str, Any]) -> tuple[str | dict[str, Any] | None, dict[str, Any] | None]:
    """One /ws/chat frame -> ``(turn, reply)``. Exactly one of the two is non-None, or both are None
    (nothing to do). ``turn`` is a prompt string, or a dict ``{"interrupt_id", "response"}`` answering
    a pending motion confirm.
    """
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(int(message.get("code") or 1000))
    raw = message.get("text")
    if raw is None:
        # bytes-only frame: a Blob/ArrayBuffer sent by mistake must not kill
        # the operator's chat with a server-side KeyError.
        return None, {"type": "error", "error": "binary frames are not accepted on /ws/chat - send JSON text"}
    if len(raw.encode("utf-8", "ignore")) > CHAT_MAX_FRAME_BYTES:
        return None, {
            "type": "error",
            "error": f"frame exceeds {CHAT_MAX_FRAME_BYTES // 1024} KB - not forwarded to the model",
        }
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None, {"type": "error", "error": 'frame is not JSON - send {"type": "chat", "text": "..."}'}
    if not isinstance(msg, dict):
        return None, {"type": "error", "error": "frame must be a JSON object"}
    if msg.get("type") == "ping":
        return None, {"type": "pong"}
    if msg.get("type") == "interrupt_response":
        iid = msg.get("id")
        if not isinstance(iid, str) or not iid.strip():
            return None, {"type": "error", "error": 'interrupt_response requires a string "id"'}
        if "response" not in msg:
            return None, {"type": "error", "error": 'interrupt_response requires a "response" field'}
        return {"interrupt_id": iid.strip(), "response": msg.get("response")}, None
    ftype = msg.get("type")
    if ftype is not None and ftype not in _CHAT_FRAME_TYPES:
        return None, {
            "type": "error",
            "error": (
                f"unknown frame type {ftype!r} on /ws/chat - nothing was done. "
                f"This server accepts: {', '.join(sorted(_CHAT_FRAME_TYPES))}."
            ),
        }
    text = msg.get("text")
    if text is None:
        return None, None
    if not isinstance(text, str):
        return None, {"type": "error", "error": f"text must be a string, got {type(text).__name__}"}
    prompt = text.strip()
    return (prompt or None), None


async def _client_gone(ws: WebSocket) -> None:
    """Park on the socket's inbound channel until the client actually leaves."""
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


def _audit_autospawn(bridge: Any, did: dict[str, Any] | None) -> None:
    """Land the auto-spawn watcher's poll results in the activity trail."""
    if not did:
        return
    for peer_id in did.get("spawned") or []:
        bridge.record_activity(
            "api",
            "spawn",
            target=peer_id,
            detail="USB auto-spawn (board plugged in)",
            ok=True,
        )
    for peer_id in did.get("despawned") or []:
        bridge.record_activity(
            "api",
            "despawn",
            target=peer_id,
            detail="USB auto-spawn (board unplugged)",
            ok=True,
        )


_HASHED_EXT = r"(?:js|css|woff2?|png|svg|jpg|jpeg|webp|ico)"
# Q184: a HUMAN-WORD tail — TitleCase words, optionally -/_ joined, no digits. That is what a
# person writes (Logo-Wordmark, hero-BannerImage, icon-Placeholder) and what a random base64url
# hash essentially never is (Q116's digit-less BGRlFtdn has consecutive capitals, so it does not
# fit). Measured 2026-08-22: BOTH patterns below matched all three of those names — "mixed case"
# alone is a hash tell only against lowercase kebab-case, not against TitleCase. Misreading a rare
# word-shaped hash costs one 304 revalidation; a year-long immutable cache on a hand-written name
# cannot be fixed from here, so the guard errs toward no-cache.
_WORDLIKE_TAIL = r"(?:[A-Z][a-z]+)(?:[-_]?[A-Z][a-z]+)*"
# A hash with no '-' in it: a digit or mixed case is tell enough (Q116 - one vite hash in four has
# no digit at all, so "must contain a digit" refused a real main bundle), EXCEPT the word-shaped
# names Q184 carves back out.
_HASHED_NAME = re.compile(
    r".+-(?!" + _WORDLIKE_TAIL + r"\.)"
    r"(?=[A-Za-z0-9_]*(?:[0-9]|[a-z][A-Za-z0-9_]*[A-Z]|[A-Z][A-Za-z0-9_]*[a-z]))"
    r"[A-Za-z0-9_]{8,}\." + _HASHED_EXT
)
# Q177: vite hashes are base64URL, so '-' appears INSIDE the hash - index-BBNIi-aw.js, the real main
# bundle of this dist, which the charset above cannot parse. A hyphen is also what hand-written
# kebab-case names are made of, so this variant demands MIXED CASE plus (Q184) a digit OR an
# internal -/_ flanked by alphanumerics, and refuses the word-shaped tails outright:
# camera-preview-2x.png and apple-touch-icon-192.png are lowercase+digit and stay revalidated,
# Logo-Wordmark.png is mixed-case but word-shaped and stays revalidated too. This pattern is a BELT
# for a call site that loses the directory: both current call sites keep it (server passes the full
# file path / URL sub-path), so assets/* is already immutable via the directory rule above.
_HASHED_BASE64URL = re.compile(
    r".+-(?!" + _WORDLIKE_TAIL + r"\.)"
    r"(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])"
    r"(?=[A-Za-z0-9_-]*(?:[0-9]|[A-Za-z0-9][-_][A-Za-z0-9]))"
    r"[A-Za-z0-9_-]{8,}\." + _HASHED_EXT
)


def static_cache_control(path: str) -> str:
    """Cache-Control for one built-frontend file, as a pure function of its name."""
    name = path.rsplit("/", 1)[-1]
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2] == "assets":
        return "public, max-age=31536000, immutable"
    if _HASHED_NAME.fullmatch(name) or _HASHED_BASE64URL.fullmatch(name):
        return "public, max-age=31536000, immutable"
    # Entry points: the html, the service worker, its registration shim, the manifest. Getting any of
    # these from a cache pins the whole app at an old build.
    return "no-cache"


def create_app(bridge: MeshBridge | None = None) -> FastAPI:
    app = FastAPI(title="strands-robots dashboard")
    from strands_robots.dashboard.config_api import load_env_file

    exported, shadowed = load_env_file()
    if exported:
        logger.info("loaded %d keys from .env: %s", len(exported), ", ".join(exported))
    if shadowed:
        logger.warning(
            ".env keys ignored because this process was launched with a different value: %s "
            "(the launch environment wins; the settings screen marks them)",
            ", ".join(shadowed),
        )
    origins = settings.get("security", "cors_origins", []) or []
    if "*" in [str(o) for o in origins]:
        logger.warning(
            "security.cors_origins contains '*': any site may READ this API with a "
            "valid token. Cross-origin writes and websockets stay refused regardless "
            "(a wildcard is not a write permit) - name the origins you actually use."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Credentialed wildcard CORS is rejected by browsers anyway, and the
        # dashboard authenticates with a bearer token, not cookies.
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TokenAuthMiddleware)

    app.state.bridge = bridge or MeshBridge()
    app.state.camera_close_log = CloseLogThrottle()
    app.state.refusals = RefusalTally()
    app.state.camera_churn = ChurnGuard()
    app.state.mesh_online = False
    app.state.devices = DeviceManager()
    # /api/record - teleop episode recording (record screen). The controller
    # parks the arms' fleet peers around a session; see record_api.py.
    from strands_robots.dashboard import record_api

    app.state.record = record_api.RecordController(app.state.devices, bridge=app.state.bridge)
    # Late-bound on purpose: capturing the bound method here would pin the
    # router to THIS bridge instance forever (tests swap it, restart_mesh may).
    app.include_router(
        record_api.build_router(
            app.state.record,
            on_activity=lambda *a, **k: app.state.bridge.record_activity(*a, **k),
        )
    )
    # A peer with a LIVE managed local process is never aged out of the fleet
    # snapshot, even if its state stream goes quiet.
    app.state.bridge.protected_peer_ids = lambda: {
        pid for pid, m in list(app.state.devices.robots.items()) if m.alive()
    }
    app.state.bridge.peer_annotations = app.state.devices.annotations_by_peer
    app.state.bridge.managed_children = app.state.devices.managed_children

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        app.state.mesh_online = await asyncio.to_thread(app.state.bridge.start, loop)
        # Hand the mesh gateway to the fleet agent (chat + voice).
        from strands_robots.dashboard import agent_bridge as _ab

        _ab.set_bridge(app.state.bridge)
        # ...and the device roster, so a direct-serial refusal can NAME the
        # spawned child holding the bus instead of printing a bare pid.
        _ab.set_devices(app.state.devices)

        # USB auto-spawn: a board with a saved profile comes up on its own,
        # and an unplugged one is stopped. Unknown boards are only reported.
        watcher = app.state.devices.start_autospawn(
            peer_ids=lambda: list(app.state.bridge.peers),
        )
        if watcher is not None:
            app.state.autospawn_task = asyncio.create_task(_autospawn_loop(watcher))

    async def _autospawn_loop(watcher: Any) -> None:
        from strands_robots.dashboard.device_manager import AUTOSPAWN_POLL_S

        while True:
            try:
                did = await asyncio.to_thread(watcher.poll)
                _audit_autospawn(app.state.bridge, did)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A blind watcher is worse than a loud one: say it, keep polling.
                logger.warning("USB auto-spawn poll failed: %r", e)
            await asyncio.sleep(AUTOSPAWN_POLL_S)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task = getattr(app.state, "autospawn_task", None)
        if task is not None:
            task.cancel()
        app.state.devices.shutdown()
        await asyncio.to_thread(app.state.bridge.stop)

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------

    @app.get("/api/network/hint")
    async def network_hint(request: Request) -> dict[str, Any]:
        fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
        client_ip = (fwd.split(",")[0].strip() if fwd else None) or (request.client.host if request.client else None)
        own: list[str] = []
        try:  # psutil is already a dashboard dependency; a failure here is not fatal
            import psutil

            for addrs in psutil.net_if_addrs().values():
                own.extend(a.address for a in addrs if a.address)
        except Exception:  # pragma: no cover - platform specific
            pass
        return lan_hint.hint(client_ip, own, int(getattr(app.state, "port", None) or 8090))

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        # Q178: mesh_ingest publishes the numbers that can CONTRADICT mesh_online (freshest
        # presence age + fan-out since the last poll). mesh_online stays for compatibility, but it
        # is the bridge's belief; `mesh.verdict` is the falsifiable reading.
        from strands_robots.dashboard.health_ingest import mesh_ingest
        from strands_robots.dashboard.mesh_bridge import PEER_STALE_S

        _coalesce = app.state.bridge.coalesce_stats()
        _ingest, app.state.mesh_ingest_prev = mesh_ingest(
            app.state.bridge.peers,
            _coalesce,
            time.time(),
            getattr(app.state, "mesh_ingest_prev", None),
            stale_after=PEER_STALE_S,
        )
        return {
            "status": "ok",
            "mesh": _ingest,
            "mesh_online": app.state.mesh_online,
            "dashboard_peer_id": app.state.bridge.peer_id,
            "peers": len(app.state.bridge.peers),
            # How much /ws/mesh fan-out the coalescer avoided.
            "mesh_coalesce": _coalesce,
            **({"joint_streams": js} if (js := silent_arms(app.state.bridge.peers)) is not None else {}),
            # Which build is answering.
            "build": build_info(),
            "t": time.time(),
            **(
                {"refused_handshakes": s}
                if (
                    s := app.state.refusals.summary(
                        time.time(),
                        # /api/health is public BY DESIGN (the caretaker polls it, the LAN hint needs it before any
                        # sign-in), so the identities in this block are withheld from a caller who has not
                        # authenticated.
                        detailed=_health_reader_is_trusted(request),
                    )
                )
                is not None
                else {}
            ),
        }

    @app.get("/api/fleet")
    async def fleet() -> dict[str, Any]:
        # Roles ride along inside snapshot() (bridge.peer_annotations), so this
        # route and the WS stream cannot disagree about which arm is the leader.
        return cast("dict[str, Any]", app.state.bridge.snapshot())

    # ------------------------------------------------------------------
    # WebAuthn passkey auth (see dashboard/auth.py)
    # ------------------------------------------------------------------

    def _require_auth_module() -> Any:
        if dash_auth is None:
            raise HTTPException(503, "passkey auth unavailable - install webauthn + PyJWT")
        return dash_auth

    def _session_presented(request: Request) -> str:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return request.query_params.get("token", "").strip()

    def _health_reader_is_trusted(request: Request) -> bool:
        """May this caller see WHO is being refused?"""
        presented = _session_presented(request)
        token = settings.get("security", "auth_token")
        if token and hmac.compare_digest(presented, str(token)):
            return True
        if dash_auth is not None and dash_auth.session_is_valid(presented):
            return True
        passkeys_on = dash_auth is not None and dash_auth.auth_enabled()
        if not token and not passkeys_on:
            client = request.client.host if request.client else None
            if client in (None, "testclient"):
                return True
            if dash_auth is not None:
                return bool(dash_auth.client_is_loopback(client))
            return client in ("127.0.0.1", "::1", "localhost")
        return False

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        out = mod.status(request)
        out["authenticated"] = mod.session_is_valid(_session_presented(request))
        return cast("dict[str, Any]", out)

    @app.post("/api/auth/register/begin")
    async def auth_register_begin(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json() if await request.body() else {}
        if mod.has_credentials() and not mod.session_is_valid(_session_presented(request)):
            raise HTTPException(401, "enrolling another passkey requires a signed-in session")
        return cast(
            "dict[str, Any]",
            mod.begin_registration(
                request,
                label=str(body.get("label") or "passkey")[:64],
                bootstrap=str(body.get("bootstrap") or ""),
            ),
        )

    @app.post("/api/auth/register/finish")
    async def auth_register_finish(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json()
        first_time = not mod.has_credentials()
        if not first_time and not mod.session_is_valid(_session_presented(request)):
            raise HTTPException(401, "enrolling another passkey requires a signed-in session")
        return cast(
            "dict[str, Any]",
            mod.finish_registration(request, body.get("challenge_id", ""), body.get("credential") or {}),
        )

    @app.post("/api/auth/login/begin")
    async def auth_login_begin(request: Request) -> dict[str, Any]:
        return cast("dict[str, Any]", _require_auth_module().begin_authentication(request))

    @app.post("/api/auth/login/finish")
    async def auth_login_finish(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json()
        return cast(
            "dict[str, Any]",
            mod.finish_authentication(request, body.get("challenge_id", ""), body.get("credential") or {}),
        )

    @app.get("/api/auth/credentials")
    async def auth_credentials() -> dict[str, Any]:
        # reached only with a valid session (guarded path, not in PUBLIC_PATHS)
        return {"credentials": _require_auth_module().list_credentials()}

    @app.delete("/api/auth/credentials/{cred_id}")
    async def auth_credential_delete(cred_id: str) -> dict[str, Any]:
        return cast("dict[str, Any]", _require_auth_module().delete_credential(cred_id))

    @app.post("/api/auth/handoff")
    async def auth_handoff(request: Request) -> dict[str, Any]:
        """A short-lived token that carries THIS signed-in session to the LAN address in a
        URL (plain-http LAN = no WebAuthn, so the passkey cannot follow). Guarded path:
        the middleware admitted a static token, a valid session, or an open loopback."""
        mod = _require_auth_module()
        presented = _session_presented(request)
        static = settings.get("security", "auth_token")
        if static and hmac.compare_digest(presented, str(static)):
            # A static token holds no claims; mint a fresh short session for the trip.
            now = int(time.time())
            ttl = mod.handoff_ttl()
            return {
                "token": mod.issue_token("static-token", name="handoff", exp=now + ttl, via="handoff"),
                "exp": now + ttl,
                "expires_in": ttl,
            }
        if not mod.session_is_valid(presented):
            # Open-loopback mode: the LAN page opens without a token too — say so
            # instead of minting a credential nobody needs in a URL.
            return {"token": None, "why": "auth is not enabled here - the local address opens without a token"}
        return cast("dict[str, Any]", mod.issue_handoff(mod.verify_token(presented)))

    @app.get("/api/robots/registry")
    async def registry() -> dict[str, Any]:
        from strands_robots.registry import list_robots

        try:
            robots = list_robots()
        except Exception as exc:  # registry read is best-effort
            raise HTTPException(500, f"registry unavailable: {exc}") from exc
        return {"robots": robots}

    @app.get("/api/policies")
    async def policies() -> dict[str, Any]:
        """Full provider catalog - the schema the run form is generated from."""
        catalog = await asyncio.to_thread(config_api._policy_catalog)
        if not catalog:
            raise HTTPException(500, "policy registry unavailable")
        return {"providers": catalog, "names": [p["name"] for p in catalog]}

    def require_peer(peer_id: str) -> None:
        """404 for a peer that was never in the fleet, before spending the RPC."""
        from strands_robots.dashboard.mesh_bridge import peer_is_known

        bridge = app.state.bridge
        managed = getattr(getattr(app.state, "devices", None), "robots", None) or {}
        if peer_is_known(peer_id, getattr(bridge, "peers", None) or {}, managed):
            return
        known = sorted(set(getattr(bridge, "peers", None) or {}) | set(managed))
        raise HTTPException(
            404,
            {
                "error": f"no peer {peer_id!r} in the fleet",
                "hint": "GET /api/fleet lists the peers that can be commanded",
                "known_peers": known,
            },
        )

    @app.get("/api/robots/{peer_id}/teleop")
    async def teleop_status(peer_id: str) -> dict[str, Any]:
        """Live teleop health for one peer: publisher/receiver rates, drops, slew rejections - the counters
        InputPublisher/InputReceiver already keep.
        """
        require_peer(peer_id)
        result = await app.state.bridge.send_cmd_async(peer_id, {"action": "teleop_status"}, timeout=10.0)
        # The counters alone lied: a follower refusing EVERY frame reported running:true, and the
        # reason existed only in its child log. health turns the counters (+ that log, when the peer
        # is ours) into a sentence, and a refusal we can continue from arrives with its consent
        # request attached.
        inner = result.get("result") if isinstance(result, dict) else None
        log_tail = None
        managed = app.state.devices.robots.get(peer_id)
        if managed is not None:
            log_tail = list(getattr(managed, "logs", []) or [])[-40:]
        health = teleop_health(inner if inner is not None else result, log_tail)
        # "Nothing is arriving" is not yet an answer: it could be a leader that never started, or two
        # peers that are not meeting.
        silent = {k: v for k, v in health.get("receivers", {}).items() if v.get("state") == "silent"}
        if silent:
            counted: dict[str, int] = {}
            for key in silent:
                source, _, device = key.partition("/")
                if not source or source == peer_id:
                    continue
                try:
                    src = await app.state.bridge.send_cmd_async(source, {"action": "teleop_status"}, timeout=10.0)
                except Exception as exc:  # noqa: BLE001 - a quiet leader is data too
                    logger.debug("could not ask leader %s about its publisher: %r", source, exc)
                    continue
                frames = published_frames(src.get("result") if isinstance(src, dict) else src, device or "leader")
                if frames is not None:
                    counted[key] = frames
            if counted:
                health = teleop_health(inner if inner is not None else result, log_tail, counted)
        worst = health.get("worst") or {}
        if worst.get("state") == "refusing" and log_tail:
            for line in reversed(log_tail):
                request = consent.classify_refusal(str(line))
                if request is not None:
                    health["needs_consent"] = request.as_dict()
                    break
        return {"peer_id": peer_id, "result": result, "health": health}

    @app.post("/api/robots/{peer_id}/teleop/publish")
    async def teleop_publish(peer_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        require_peer(peer_id)
        body = body or {}
        cmd: dict[str, Any] = {"action": "teleop_publish"}
        for field in ("device_name", "hz", "robot_name"):
            if body.get(field) is not None:
                cmd[field] = body[field]
        result = await app.state.bridge.send_cmd_async(peer_id, cmd, timeout=30.0)
        return {"peer_id": peer_id, "result": result}

    @app.post("/api/robots/{peer_id}/teleop/receive")
    async def teleop_receive(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Point a follower (real or sim twin) at a leader's input stream.

        With the TeleopMixin lift a SIM twin can follow a REAL leader arm -
        practice-on-the-twin before metal.
        """
        require_peer(peer_id)
        source = (body.get("source_peer_id") or "").strip()
        if not source:
            raise HTTPException(422, "source_peer_id required")
        # The leader is a peer too: pointing a follower at a stream nobody
        # publishes is a 45s wait ending in a shrug.
        require_peer(source)
        cmd = {
            "action": "teleop_receive",
            "source_peer_id": source,
            "device_name": body.get("device_name", "leader"),
        }
        # First declare_subscriber on a peer can take >15s (zenoh declare +
        # gossip propagation) - not a deadlock, just slow. 45s budget.
        result = await app.state.bridge.send_cmd_async(peer_id, cmd, timeout=45.0)
        return {"peer_id": peer_id, "result": result}

    @app.post("/api/robots/{peer_id}/teleop/stop")
    async def teleop_stop(peer_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        require_peer(peer_id)
        cmd: dict[str, Any] = {"action": "teleop_stop"}
        if body and body.get("device_name"):
            cmd["device_name"] = body["device_name"]
        result = await app.state.bridge.send_cmd_async(peer_id, cmd, timeout=10.0)
        return {"peer_id": peer_id, "result": result}

    @app.post("/api/collect")
    async def collect_episodes(body: dict[str, Any]) -> dict[str, Any]:
        """Collect a policy-driven dataset in a one-shot mesh sim. run_policy drives exactly n_episodes
        rollouts with per-episode parquet boundaries and reports parquet-truth counts.
        """
        dataset_root = (body.get("dataset_root") or "").strip()
        if not dataset_root:
            raise HTTPException(422, "dataset_root required")
        # Remember the root so /api/training/datasets discovers the result
        # even outside the default scan paths (HF_LEROBOT_HOME etc.).
        from strands_robots.dashboard import training as _training

        _training.remember_dataset_root(dataset_root)
        result = await asyncio.to_thread(
            lambda: app.state.devices.collect(
                dataset_root=dataset_root,
                dataset_repo_id=body.get("dataset_repo_id", "local/collected"),
                robot_name=body.get("robot_name") or "so101",
                policy_provider=body.get("policy_provider", "mock"),
                policy_config=body.get("policy_config"),
                instruction=body.get("instruction", ""),
                n_episodes=int(body.get("n_episodes", 5)),
                duration=float(body.get("duration", 10.0)),
                fps=int(body.get("fps", 30)),
            )
        )
        # Two recorders writing one dataset directory interleave episodes into
        # each other's files. 409 names the session already holding it.
        if result.get("already_running"):
            raise HTTPException(409, result)
        return cast("dict[str, Any]", result)

    @app.post("/api/replay")
    async def replay_episode(body: dict[str, Any]) -> dict[str, Any]:
        """Replay a recorded LeRobotDataset episode in a one-shot mesh sim."""
        repo_id = (body.get("repo_id") or "").strip()
        if not repo_id:
            raise HTTPException(422, "repo_id required")
        from strands_robots.dashboard.device_manager import validate_replay

        bad = validate_replay(repo_id, body.get("episode", 0), body.get("root"), body.get("speed", 1.0))
        if bad:
            raise HTTPException(422, bad)
        result = await asyncio.to_thread(
            app.state.devices.replay,
            repo_id,
            int(body.get("episode", 0)),
            body.get("root"),
            float(body.get("speed", 1.0)),
            body.get("robot_name") or "so101",
        )
        # 409, not an error-shaped 200: a second replay of the same episode is a
        # conflict with something that already exists, and the response names
        # the peer whose card is already showing it.
        if result.get("already_running"):
            raise HTTPException(409, result)
        return cast("dict[str, Any]", result)

    @app.get("/api/training/trainers")
    async def training_trainers() -> dict[str, Any]:
        from strands_robots.dashboard import training

        # `trainers` keeps its shape (a list of names) forever: this app ships as a PWA and a cached
        # older bundle renders that list directly - changing it to objects crashes those tabs.
        return {
            "trainers": await asyncio.to_thread(training.list_trainers),
            "unsupported": await asyncio.to_thread(training.form_unsupported),
            "fields": list(training.SPEC_KEYS),
        }

    @app.get("/api/datasets/labels")
    async def dataset_labels(root: str) -> dict[str, Any]:
        from pathlib import Path

        from strands_robots import episode_labels as _labels
        from strands_robots.dashboard.episode_label_view import label_view

        target = Path(root).expanduser()
        if not target.is_dir():
            raise HTTPException(404, f"no dataset directory at {target}")

        document: dict[str, Any] | None = None
        sidecar_error: str | None = None
        if _labels.labels_path(target).exists():
            try:
                document = _labels.read_labels(target)
            except Exception as e:  # a corrupt sidecar must not read as "no labels yet"
                sidecar_error = f"{type(e).__name__}: {e}"

        total: int | None = None
        try:
            import json as _json

            total = _json.loads((target / "meta" / "info.json").read_text()).get("total_episodes")
        except Exception:  # noqa: BLE001 - a dataset mid-recording has no readable info.json yet
            pass

        return label_view(document, total_episodes=total, sidecar_error=sidecar_error)

    @app.get("/api/training/datasets")
    async def training_datasets(q: str = "", hub: bool = True, limit: int = 12) -> dict[str, Any]:
        """Datasets for the submit form's picker: local roots + a Hub search."""
        from strands_robots.dashboard import training
        from strands_robots.dashboard.dataset_check import mark_live_recording

        active, captured = None, None
        try:
            session = getattr(app.state, "record", None)
            live = session.session() if session is not None else {}
            active = live.get("dataset") or None
            episodes = live.get("episodes")
            captured = len(episodes) if isinstance(episodes, list) else None
        except Exception:  # noqa: BLE001 - the picker is worth more than this annotation
            active, captured = None, None

        if not hub:
            rows = await asyncio.to_thread(training.local_datasets, q)
            return {"datasets": mark_live_recording(rows, active, episodes_so_far=captured)}
        from strands_robots.dashboard import checkpoints

        found = await asyncio.to_thread(training.search_datasets, q, checkpoints.clamp_limit(limit, 12, 50))
        return {**found, "datasets": mark_live_recording(found.get("datasets", []), active, episodes_so_far=captured)}

    @app.get("/api/training/jobs")
    async def training_jobs() -> dict[str, Any]:
        from strands_robots.dashboard import training

        rows = await asyncio.to_thread(training.jobs)
        # `problem` is about the LEDGER, not the runs: an unreadable history and a
        # dashboard that has never trained anything both produce an empty list, and
        # only one of them means runs were forgotten.
        return {"jobs": rows, "problem": training.jobs_problem()}

    @app.post("/api/training/validate")
    async def training_validate(body: dict[str, Any]) -> dict[str, Any]:
        from strands_robots.dashboard import training

        return await asyncio.to_thread(training.validate, body)

    @app.get("/api/training/output-dir")
    async def training_output_dir(path: str = "") -> dict[str, Any]:
        from strands_robots.dashboard import training

        if not path.strip():
            raise HTTPException(422, "path is required")
        return await asyncio.to_thread(training.output_dir_verdict, path.strip())

    @app.post("/api/training/submit")
    async def training_submit(body: dict[str, Any]) -> dict[str, Any]:
        """Launch a training job (train_policy validates the spec first)."""
        from strands_robots.dashboard import training

        result = await asyncio.to_thread(training.submit, body)
        job = (result.get("data") or {}).get("job_id") if isinstance(result, dict) else None
        app.state.bridge.record_activity(
            "training",
            "submit",
            target=str(job or body.get("provider", "?")),
            detail=f"{body.get('provider')} on {body.get('dataset_root') or body.get('dataset_repo_id') or '?'}",
            ok=bool(isinstance(result, dict) and result.get("status") == "success"),
        )
        return result

    @app.get("/api/training/status")
    async def training_status(provider: str, job_id: str) -> dict[str, Any]:
        from strands_robots.dashboard import training

        return await asyncio.to_thread(training.status, provider, job_id)

    @app.post("/api/training/export")
    async def training_export(body: dict[str, Any]) -> dict[str, Any]:
        """Export the latest checkpoint as a loadable policy artifact."""
        from strands_robots.dashboard import training

        return await asyncio.to_thread(
            training.export,
            body.get("provider", "lerobot_local"),
            body.get("output_dir", ""),
            body.get("dataset_root", ""),
            body.get("dataset_repo_id"),
            body.get("base_model", ""),
        )

    @app.get("/api/checkpoints/search")
    async def checkpoints_search(q: str = "", limit: int = 15) -> dict[str, Any]:
        """Type-ahead checkpoint search for the run form. Merges the local HF cache (instant, marked
        ``local``) with a Hub search of public LeRobot checkpoints ranked by downloads.
        """
        from strands_robots.dashboard import checkpoints

        return await asyncio.to_thread(checkpoints.search, q, checkpoints.clamp_limit(limit))

    @app.get("/api/checkpoints/features")
    async def checkpoint_features(repo_id: str = "") -> dict[str, Any]:
        from strands_robots.dashboard import checkpoints

        return await asyncio.to_thread(checkpoints.declared_features, repo_id)

    @app.get("/api/checkpoints/families")
    async def checkpoint_families() -> dict[str, Any]:
        """policy_type values the lerobot family accepts (type dropdown)."""
        from strands_robots.dashboard import checkpoints

        return {"families": await asyncio.to_thread(checkpoints.policy_families)}

    @app.post("/api/policies/validate")
    async def validate_policy(body: dict[str, Any]) -> dict[str, Any]:
        """Dry-run a provider config without touching a robot."""
        provider = (body.get("policy_provider") or "").strip()
        if not provider:
            raise HTTPException(422, "policy_provider required")
        config = body.get("policy_config") or {}
        if not isinstance(config, dict):
            raise HTTPException(422, "policy_config must be an object")

        # Preflight's most useful check compares the model's declared image inputs against the
        # observation keys it will actually receive, so pass the target peer's real joints + cameras
        # when we know them.
        peer_id = body.get("peer_id") or ""
        peer = app.state.bridge.peers.get(peer_id) or {}
        observation_keys = set((peer.get("state") or {}).get("joints") or {})
        observation_keys |= set(peer.get("cameras") or {})

        def _check() -> dict[str, Any]:
            from strands_robots.policies import policy_provider_error, preflight_policy

            problem = policy_provider_error(provider, **config)
            if problem:
                return {"ok": False, "stage": "provider", "error": problem}
            try:
                from strands_robots.policies.factory import _check_trust_remote_code

                _check_trust_remote_code(provider)
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001 - the trust gate, verbatim
                return {"ok": False, "stage": "trust", "error": str(exc)}
            try:
                # Returns None and raises ValueError on a bad config; a
                # provider without a preflight hook is a silent pass.
                preflight_policy(provider, observation_keys, **config)
            except Exception as exc:  # noqa: BLE001 - surfacing is the point
                return {"ok": False, "stage": "preflight", "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "stage": "preflight"}

        result = await asyncio.to_thread(_check)
        result["policy_provider"] = provider
        result["observation_keys"] = sorted(observation_keys)
        if result.get("ok"):
            from strands_robots.dashboard.validate_scope import validation_scope

            spec = None
            try:
                catalog = await asyncio.to_thread(config_api._policy_catalog)
                spec = next((p for p in (catalog or []) if p.get("name") == provider), None)
            except Exception:  # noqa: BLE001 - a scope hint must never break validate
                spec = None
            scope = validation_scope(spec, config)
            result["resolved"] = scope["resolved"]
            if scope["scope_note"]:
                result["scope_note"] = scope["scope_note"]
        if not observation_keys:
            result["note"] = "no live observation keys for that peer - camera/joint routing was not checked"

        try:
            from strands_robots.dashboard.checkpoints import declared_features
            from strands_robots.dashboard.policy_fit import policy_fit

            ckpt = next(
                (
                    str(config[k])
                    for k in ("pretrained_name_or_path", "model_path", "checkpoint")
                    if isinstance(config.get(k), str) and config.get(k, "").strip()
                ),
                "",
            )
            if ckpt:
                feats = await asyncio.to_thread(declared_features, ckpt)
                if feats:
                    fit = policy_fit(
                        input_features=feats.get("input_features"),
                        output_features=feats.get("output_features"),
                        joints=list((peer.get("state") or {}).get("joints") or {}),
                        cameras=list(peer.get("cameras") or {}),
                        norm_tag=config.get("norm_tag") if isinstance(config.get("norm_tag"), str) else None,
                        declared_norm_tags=feats.get("norm_tags"),
                        # runRisk's server-side twin: `hw` exists only when a real device object is
                        # attached, and an unknown peer is treated as metal.
                        physical=bool(peer.get("presence", {}).get("hw")) or not peer,
                    )
                    result["fit"] = fit
                    if fit.get("blocking"):
                        # A fit problem is not a config typo: it is the wrong policy for this robot,
                        # and no amount of correcting fields makes it run. It overrides `ok`, because
                        # the form arms play on that flag.
                        result["ok"] = False
                        result["stage"] = "fit"
                        result["error"] = "; ".join(p.get("detail", "") for p in fit.get("problems", []))
        except Exception:  # noqa: BLE001 - a fit hint must never break validate
            logger.debug("policy fit check failed", exc_info=True)

        return result

    @app.get("/api/robots/{peer_id}/policy-fit")
    async def policy_fit_route(peer_id: str, repo_id: str = "", norm_tag: str = "") -> dict[str, Any]:
        require_peer(peer_id)
        from strands_robots.dashboard.checkpoints import declared_features
        from strands_robots.dashboard.policy_fit import policy_fit

        peer = app.state.bridge.peers.get(peer_id) or {}
        joints = list((peer.get("state") or {}).get("joints") or {})
        cameras = list(peer.get("cameras") or {})
        feats = await asyncio.to_thread(declared_features, repo_id) if repo_id.strip() else {}
        verdict = policy_fit(
            input_features=feats.get("input_features"),
            output_features=feats.get("output_features"),
            joints=joints,
            cameras=cameras,
            physical=bool((peer.get("presence") or {}).get("hw")) or not peer,
            norm_tag=norm_tag,
            declared_norm_tags=feats.get("norm_tags"),
        )
        verdict["evidence"] = bool(feats) and bool(verdict["checked"])
        verdict["repo_id"] = repo_id
        verdict["policy_type"] = feats.get("policy_type")
        verdict["robot"] = {"joints": joints, "cameras": cameras}
        return verdict

    @app.post("/api/robots/{peer_id}/task")
    async def start_task(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        require_peer(peer_id)
        instruction = (body.get("instruction") or "").strip()
        if not instruction:
            raise HTTPException(422, "instruction required")
        # An opt-in anti-accident lock (off unless the operator set it): a task POST that would start
        # REAL motion must carry the browser's confirmation. play sends it; a curl against the public
        # tunnel does not.
        from strands_robots.dashboard.agent_motion import task_post_allowed

        verdict = task_post_allowed(
            peer=app.state.bridge.peers.get(peer_id),
            confirmed=bool(body.get("confirmed")),
            target=peer_id,
        )
        if not verdict["allowed"]:
            raise HTTPException(403, verdict["reason"])
        duration = float(body.get("duration", 30.0))
        cmd = {
            # Sim peers accept "start"; hardware peers accept both. "execute"
            # is kept as an explicit opt-in for callers that need it.
            "action": body.get("action") or "start",
            "instruction": instruction,
            "policy_provider": body.get("policy_provider", "mock"),
            "duration": duration,
        }
        # Only wire-settable keys: validate_command() builds its output from a strict per-action
        # allowlist and *silently drops* everything else, so forwarding e.g. a policy_config dict
        # would look accepted and arrive empty. config_api.WIRE_CMD_KEYS is that allowlist.
        for opt in config_api.WIRE_CMD_KEYS:
            if body.get(opt) is not None:
                cmd[opt] = body[opt]
        # Child sim peers can't execute themselves: route "<parent>__<robot>" to the parent with
        # robot_name here, so the card's Run button and every API caller get the fix - not just the
        # agent's fleet tool.
        from strands_robots.dashboard.mesh_bridge import route_task_target

        target, cmd = route_task_target(peer_id, cmd)
        # Two different waits: "start" is answered by an immediate ack (so waiting duration+10 meant a
        # 1-hour run held Run in "starting" for 3610s if the peer never answered), while "execute"
        # blocks until the rollout ends.
        from strands_robots.dashboard.task_timeout import task_ack_budget, timeout_verdict

        timeout_s, timeout_kind = task_ack_budget(cmd["action"], body.get("timeout"), duration)
        # Twin mirroring: fire the same instruction at '<peer>-twin' when one
        # is live (fire-and-forget; twin progress streams on the mesh).
        twin_id = f"{peer_id}-twin"
        twin = app.state.bridge.peers.get(twin_id)
        mirrored = bool(twin and not twin.get("stale"))
        if mirrored:
            t_target, t_cmd = route_task_target(twin_id, dict(cmd))
            asyncio.create_task(app.state.bridge.send_cmd_async(t_target, t_cmd, timeout=timeout_s))
        result = await app.state.bridge.send_cmd_async(target, cmd, timeout=timeout_s)
        from strands_robots.dashboard.mesh_bridge import command_succeeded

        payload = {
            "peer_id": peer_id,
            "routed_to": target if target != peer_id else None,
            "mirrored_to_twin": mirrored,
            # A response can arrive and still say ok=False. The UI needs one
            # honest boolean, not a nested shape it has to re-guess.
            "ok": command_succeeded(result),
            "timeout_s": timeout_s,
            "result": result,
        }
        # A timeout is not "nothing happened": the command was delivered, so the
        # robot may be loading a policy and about to move. Say which wait ended.
        if isinstance(result, dict) and not payload["ok"] and str(result.get("error", "")).startswith("timeout"):
            verdict = timeout_verdict(timeout_kind, timeout_s, target)
            result.update(verdict)
            payload["motion_possible"] = True
            payload["timeout_kind"] = verdict["timeout_kind"]
        if not payload["ok"] and isinstance(result, dict):
            # The refusal is inside the peer's answer, under whichever key that
            # peer chose; check the ones that carry prose rather than guessing one.
            consent.attach_consent(
                payload,
                result.get("error"),
                result.get("detail"),
                result.get("message"),
                result.get("reason"),
            )
        return payload

    @app.post("/api/robots/{peer_id}/stop")
    async def stop_task(peer_id: str) -> dict[str, Any]:
        require_peer(peer_id)
        result = await app.state.bridge.send_cmd_async(peer_id, {"action": "stop"}, timeout=10.0)
        outcome = stop_outcome(result)
        return {"peer_id": peer_id, **outcome, "result": result}

    @app.post("/api/safety/estop")
    async def estop() -> dict[str, Any]:
        """Fleet-wide stop: BOTH rails fire, results reported side by side."""
        bridge: MeshBridge = app.state.bridge
        peers = bridge.live_peers()
        stale = sorted(set(bridge.peers) - set(peers))
        results = await asyncio.gather(
            *(bridge.send_cmd_async(p, {"action": "stop"}, timeout=5.0, source="estop") for p in peers),
            return_exceptions=True,
        )
        per_peer: dict[str, dict[str, Any]] = {}
        for peer, raw in zip(peers, results):
            result = raw if isinstance(raw, dict) else {"error": str(raw)}
            per_peer[peer] = {**stop_outcome(result), "result": result}
        counts = {"stopped": 0, "not_stopped": 0, "no_answer": 0}
        for info in per_peer.values():
            counts[info["state"]] = counts.get(info["state"], 0) + 1
        bridge.record_activity(
            "estop",
            "stop_all",
            target="fleet",
            detail=f"{counts['stopped']}/{len(peers)} confirmed stopped",
            ok=counts["stopped"] == len(peers) and bool(peers),
        )
        # A6: fire the SIGNED safety rail too - the envelope engages the fleet-wide LOCKOUT on every
        # listening peer (they refuse all further commands until a proofed resume), which per-peer
        # stop commands cannot do.
        signed = await asyncio.to_thread(bridge.signed_estop)
        return {
            "targeted": peers,
            "stale_skipped": stale,
            "counts": counts,
            # True only when every live peer confirmed. Anything else and the
            # UI must keep shouting.
            "all_stopped": bool(peers) and counts["stopped"] == len(peers),
            "stopped": per_peer,
            "signed_rail": {k: v for k, v in signed.items() if k != "responses"},
            "lockout_engaged": bool(signed.get("lockout_engaged")),
        }

    @app.post("/api/safety/resume")
    async def safety_resume(body: dict[str, Any]) -> dict[str, Any]:
        """Clear the fleet e-stop lockout with the operator override code."""
        code = (body.get("override_code") or "").strip()
        if not code:
            raise HTTPException(422, "override_code required")
        result = await asyncio.to_thread(app.state.bridge.signed_resume, code)
        bridge2: MeshBridge = app.state.bridge
        bridge2.record_activity(
            "resume",
            "safety_resume",
            target="fleet",
            detail=result.get("status", result.get("error", "?")),
            ok=result.get("status") == "ok",
        )
        return cast("dict[str, Any]", result)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        from strands_robots.dashboard.agent_bridge import agent_status

        return await asyncio.to_thread(
            lambda: config_api.snapshot(bridge=app.state.bridge, agent_status=agent_status())
        )

    @app.post("/api/config")
    async def post_config(body: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(config_api.apply, body)
        if result["errors"] and not result["applied"] and not result["env_written"]:
            raise HTTPException(422, "; ".join(result["errors"]))
        if result["restart_required"] and body.get("restart_mesh"):
            result["mesh_restart"] = await _restart_mesh(force=bool(body.get("force")))
        return result

    @app.get("/api/consent")
    async def get_consent() -> dict[str, Any]:
        return {**consent.granted_state(os.environ), "env_file": str(config_api.ENV_FILE)}

    @app.post("/api/consent")
    async def post_consent(body: dict[str, Any]) -> dict[str, Any]:
        """Approve one refusal, by kind + subject - nothing else is settable."""
        request = consent.build_request(str(body.get("kind", "")), body.get("subject"))
        if request is None:
            raise HTTPException(422, f"unknown consent kind; expected one of {', '.join(consent.KINDS)}")
        patch = consent.env_patch(request, os.environ)
        if not patch:
            return {
                "granted": False,
                "scope": request.scope,
                "already_granted": True,
                "note": (
                    "nothing to change - this is already allowed here. A process started "
                    "before the grant keeps the old environment: respawn it."
                ),
                "respawn_required": True,
            }
        for key, value in patch.items():
            problem = config_api.env_entry_error(key, value)
            if problem:  # pragma: no cover - the builder cannot produce one today
                raise HTTPException(422, problem)
        written = await asyncio.to_thread(config_api.upsert_env_file, patch)
        os.environ.update(patch)
        try:  # the mesh caches its parsed allowlist per value, so re-read it
            from strands_robots.mesh import security as _mesh_security

            _mesh_security._hf_repo_allowlist()
        except Exception:  # noqa: BLE001 - a cache warm-up must not fail a grant
            logger.debug("consent: allowlist warm-up failed", exc_info=True)
        app.state.bridge.record_activity(
            "api",
            "consent",
            target=request.scope,
            detail=f"approved: {', '.join(request.grants)}",
            ok=True,
        )
        return {
            "granted": True,
            "scope": request.scope,
            "kind": request.kind,
            "env_written": written,
            "grants": list(request.grants),
            # Children get the environment they were STARTED with. Say so
            # instead of letting a retry fail for a reason we already know.
            "respawn_required": True,
            "note": (
                "granted for new processes. A robot already running was started with the "
                "old environment - respawn it, then retry."
            ),
        }

    @app.post("/api/consent/revoke")
    async def revoke_consent(body: dict[str, Any]) -> dict[str, Any]:
        """Take one grant back - the other half of a promise the dialog makes."""
        request = consent.build_request(str(body.get("kind", "")), body.get("subject"))
        if request is None:
            raise HTTPException(422, f"unknown consent kind; expected one of {', '.join(consent.KINDS)}")
        patch = consent.revoke_patch(request, os.environ)
        if not patch:
            return {
                "revoked": False,
                "scope": request.scope,
                "note": "nothing to revoke - this machine does not grant that (an org-wide entry may still cover it).",
            }
        written = await asyncio.to_thread(config_api.upsert_env_file, patch)
        for key, value in patch.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        app.state.bridge.record_activity(
            "api",
            "consent",
            target=request.scope,
            detail="revoked",
            ok=True,
        )
        return {
            "revoked": True,
            "scope": request.scope,
            "env_written": written,
            "respawn_required": True,
            "note": (
                "revoked for new processes. A robot already running kept the permission it "
                "started with - respawn it to apply this."
            ),
        }

    async def _restart_mesh(*, force: bool = False) -> dict[str, Any]:
        """Re-open the mesh session against the current settings."""
        dm: DeviceManager = app.state.devices
        managed = [pid for pid, r in dm.robots.items() if r.alive()]
        if managed and not force:
            raise HTTPException(
                409,
                f"{len(managed)} locally spawned robot(s) hold the mesh session "
                f"({', '.join(managed)}). Despawn them first, or pass force=true "
                "to re-point anyway (they will keep using the old endpoints).",
            )
        settings.load(refresh=True)
        online = await asyncio.to_thread(app.state.bridge.restart)
        app.state.mesh_online = online
        return {
            "mesh_online": online,
            "orphaned": managed if force else [],
            "mesh": app.state.bridge.mesh_info(),
        }

    @app.get("/api/mesh/config")
    async def get_mesh_config() -> dict[str, Any]:
        return cast("dict[str, Any]", app.state.bridge.mesh_info())

    @app.post("/api/mesh/config")
    async def post_mesh_config(body: dict[str, Any]) -> dict[str, Any]:
        """Persist mesh endpoints and (by default) re-point the session."""
        body = body or {}
        mesh = {
            k: v
            for k, v in body.items()
            if k in ("connect", "listen", "port", "backend", "camera_hz", "policy_type_allow")
        }
        result = await asyncio.to_thread(config_api.apply, {"mesh": mesh})
        if result["errors"]:
            raise HTTPException(422, "; ".join(result["errors"]))
        changed = result["applied"] + result["restart_required"]
        if body.get("restart", True):
            result["mesh_restart"] = await _restart_mesh(force=bool(body.get("force")))
        result["changed"] = changed
        return result

    @app.post("/api/mesh/restart")
    async def restart_mesh(body: dict[str, Any] | None = None) -> dict[str, Any]:
        return await _restart_mesh(force=bool((body or {}).get("force")))

    @app.get("/api/agent/status")
    async def get_agent_status() -> dict[str, Any]:
        from strands_robots.dashboard.agent_bridge import agent_status

        return agent_status()

    @app.post("/api/agent/reset")
    async def post_agent_reset(body: dict[str, Any] | None = None) -> dict[str, Any]:
        from strands_robots.dashboard.agent_bridge import agent_status, reset_agent

        body = body or {}
        await asyncio.to_thread(reset_agent, clear_history=bool(body.get("clear_history")))
        return {"reset": True, "history_cleared": bool(body.get("clear_history")), **agent_status()}

    @app.get("/api/activity")
    async def activity(limit: int = 100) -> dict[str, Any]:
        """Recent commands and safety events, newest first."""
        return {"activity": app.state.bridge.activity_log(limit=max(1, min(limit, 300)))}

    def _live_camera_names() -> dict[str, list[str]]:
        """peer_id -> camera names the mesh has actually seen frames for."""
        snapshot = app.state.bridge.snapshot()
        return {
            peer_id: list((entry.get("cameras") or {}).keys())
            for peer_id, entry in (snapshot.get("peers") or {}).items()
        }

    @app.get("/api/devices")
    async def devices(refresh: bool = False) -> dict[str, Any]:
        """Local USB serial ports (servo buses) + cameras + managed robots."""
        # The mesh's frame bookkeeping is the evidence for "in use": a camera in a child's config that
        # never delivered a frame is assigned, not streaming, and the difference is what the operator
        # has to act on.
        return await asyncio.to_thread(app.state.devices.devices, refresh, _live_camera_names())

    @app.post("/api/devices/spawn-remembered")
    async def spawn_remembered(body: dict[str, Any]) -> dict[str, Any]:
        from strands_robots.dashboard.device_manager import respawn_payload

        port = str(body.get("port") or "").strip()
        if not port:
            raise HTTPException(422, "port required")
        profile = await asyncio.to_thread(app.state.devices.profile_for_port, port)
        payload = respawn_payload(profile, port)
        if payload.get("error"):
            raise HTTPException(404, payload["error"])
        moved = payload.pop("port_moved", None)
        # One spawn path, not two: the settle window, the consent attachment and the audit trail all
        # live in the route above, and a second copy of them is a second thing to forget to fix.
        result = await spawn(payload)
        result["respawned_from_profile"] = True
        if moved:
            # Said out loud because it is the operator's evidence that the board they are looking at
            # is the board that came up: same serial, new /dev path.
            result["port_moved"] = moved
        return result

    @app.get("/api/devices/camera/{index}/preview")
    async def camera_preview(index: int) -> Response:
        """One JPEG frame from an unclaimed camera index."""
        try:
            jpeg = await asyncio.to_thread(
                app.state.devices.preview_frame,
                index,
                _live_camera_names(),
            )
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
            raise HTTPException(503, str(e)) from e
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/devices/camera/{index}/modes")
    async def camera_modes(index: int) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                app.state.devices.probe_modes,
                index,
                _live_camera_names(),
            )
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
            raise HTTPException(503, str(e)) from e

    @app.get("/api/devices/arm-role")
    async def arm_role(port: str, model: str = "sts3215") -> dict[str, Any]:
        try:
            # Measures AND remembers (keyed by USB serial, never by /dev name - the OS reassigns those).
            return await asyncio.to_thread(app.state.devices.measure_arm_role, port, model)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001 - bus faults become HTTP, not tracebacks
            raise HTTPException(503, f"could not read {port}: {e}") from e

    @app.post("/api/devices/spawn")
    async def spawn(body: dict[str, Any]) -> dict[str, Any]:
        robot_name = body.get("robot_name")
        if not robot_name:
            raise HTTPException(422, "robot_name required")
        # An unspawnable mode or an unknown robot is a bad REQUEST, answered before any process
        # exists.
        from strands_robots.dashboard.device_manager import validate_spawn

        checked = await asyncio.to_thread(validate_spawn, robot_name, body.get("mode", "sim"))
        if isinstance(checked, dict):
            app.state.bridge.record_activity(
                "api",
                "spawn",
                target=str(robot_name),
                detail=f"refused: {checked['error']}",
                ok=False,
            )
            raise HTTPException(422, checked)
        result = await asyncio.to_thread(
            app.state.devices.spawn,
            robot_name,
            body.get("mode", "sim"),
            body.get("peer_id"),
            body.get("port"),
            body.get("cameras"),
            body.get("robot_id"),
        )
        # A pid is not a running robot.
        peer_id = result.get("peer_id")
        if peer_id and "error" not in result:
            bridge = app.state.bridge
            outcome = await asyncio.to_thread(
                app.state.devices.settle,
                peer_id,
                is_up=lambda pid: pid in (getattr(bridge, "peers", None) or {}),
            )
            result.update(outcome)
            if outcome.get("status") == "failed":
                # Surface it in the field every caller already reads, so a
                # dead spawn cannot be mistaken for a live one by any client.
                result["error"] = outcome.get("reason") or "the peer did not start"
                consent.attach_consent(result, result["error"], "\n".join(outcome.get("log_tail") or []))

        # Lifecycle lands in the audit trail: "who started this peer" is as
        # unanswerable as "who moved that arm" without it - the auto-spawn
        # watcher and the UI use this same route.
        app.state.bridge.record_activity(
            "api",
            "spawn",
            target=result.get("peer_id") or robot_name,
            detail=(
                f"{robot_name} mode={body.get('mode', 'sim')}"
                + (f" -> {result['error']}" if result.get("error") else "")
            ),
            ok="error" not in result,
        )
        if "already running" in (result.get("error") or ""):
            # A conflict with something that exists, not a bad request.
            raise HTTPException(409, result)
        if "port required" in (result.get("error") or ""):
            # Also a bad request: mode=real without a port describes no device.
            # It answered 200 with an error body, so a caller checking the status
            # code alone saw a successful spawn.
            raise HTTPException(422, result)
        return cast("dict[str, Any]", result)

    @app.get("/api/devices/profiles")
    async def device_profiles() -> dict[str, Any]:
        """Remembered USB device profiles, keyed by board serial number."""
        return {
            "profiles": app.state.devices.profiles.all(),
            "path": app.state.devices.profiles.path,
            "autospawn": getattr(app.state, "autospawn_task", None) is not None,
        }

    @app.post("/api/deploy/snippet")
    async def deploy_snippet(body: dict[str, Any], request: Request) -> dict[str, Any]:
        payload = body.get("payload")
        serial = body.get("serial")
        if serial and not payload:
            payload = app.state.devices.profiles.get(str(serial))
            if payload is None:
                raise HTTPException(404, f"no profile remembered for {serial!r}")
        if not isinstance(payload, dict):
            raise HTTPException(422, "payload object or serial required")
        hub_host = body.get("hub_host")
        hub_note = None
        if hub_host is None:
            hub_host, hub_note = deploy.hub_host_from_reached(request.url.hostname)
        result = deploy.render_snippet(
            payload,
            hub_host=hub_host or None,
            hub_note=hub_note,
            mesh_env=os.environ,
            hub_port=settings.get("mesh", "port", deploy.DEFAULT_HUB_PORT),
        )
        if "error" in result:
            raise HTTPException(422, result["error"])
        return result

    @app.post("/api/devices/despawn")
    async def despawn(body: dict[str, Any]) -> dict[str, Any]:
        peer_id = body.get("peer_id")
        if not peer_id:
            raise HTTPException(422, "peer_id required")
        result = await asyncio.to_thread(app.state.devices.despawn, peer_id)
        app.state.bridge.record_activity(
            "api",
            "despawn",
            target=peer_id,
            ok="error" not in result,
        )
        return cast("dict[str, Any]", result)

    @app.post("/api/devices/{peer_id}/cameras")
    async def reconfigure_cameras(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if "cameras" not in body:
            raise HTTPException(422, "cameras required (a mapping, or null to detach all)")
        from strands_robots.dashboard.device_manager import validate_cameras

        bad = validate_cameras(body.get("cameras"))
        if bad:
            raise HTTPException(422, bad)
        result = await asyncio.to_thread(app.state.devices.reconfigure_cameras, peer_id, body.get("cameras"))
        if "error" in result and not result.get("reconfigured"):
            app.state.bridge.record_activity(
                "api",
                "cameras",
                target=peer_id,
                detail=f"refused: {result['error']}",
                ok=False,
            )
            status = 404 if "unknown managed peer" in result["error"] else 409
            raise HTTPException(status, result)
        # Same honesty rail as spawn: a pid is not a running robot.
        bridge = app.state.bridge
        outcome = await asyncio.to_thread(
            app.state.devices.settle,
            peer_id,
            is_up=lambda pid: pid in (getattr(bridge, "peers", None) or {}),
        )
        result.update(outcome)
        if outcome.get("status") == "failed":
            result["error"] = outcome.get("reason") or "the peer did not come back"
        app.state.bridge.record_activity(
            "api",
            "cameras",
            target=peer_id,
            detail=f"respawned with {len(body.get('cameras') or {})} camera(s)"
            + (f" -> {result['error']}" if result.get("error") else ""),
            ok="error" not in result,
        )
        return cast("dict[str, Any]", result)

    @app.get("/api/devices/logs/{peer_id}")
    async def device_logs(peer_id: str) -> dict[str, Any]:
        """Child-process output for one managed robot (ring buffer)."""
        out = app.state.devices.logs(peer_id)
        if "error" in out:
            managed = sorted(getattr(app.state.devices, "robots", None) or {})
            raise HTTPException(
                404,
                {
                    "error": out["error"],
                    "hint": "only locally spawned robots keep a log ring buffer",
                    "managed_peers": managed,
                },
            )
        return cast("dict[str, Any]", out)

    @app.post("/api/robots/{peer_id}/twin")
    async def toggle_twin(peer_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Spawn/despawn a MuJoCo digital twin sim peer named '<peer>-twin'.

        Tasks started via /api/robots/<peer>/task are mirrored to a live
        twin.
        """
        body = body or {}
        twin_id = f"{peer_id}-twin"
        dm = app.state.devices
        existing = dm.robots.get(twin_id)
        if existing and existing.alive():
            return await asyncio.to_thread(dm.despawn, twin_id)
        robot_name = body.get("robot_name")
        if not robot_name:
            peer = app.state.bridge.peers.get(peer_id) or {}
            robot_name = (peer.get("presence") or {}).get("tool_name") or "so101"
        return await asyncio.to_thread(dm.spawn, robot_name, "sim", twin_id)

    @app.get("/api/calibration")
    async def calibration_list() -> dict[str, Any]:
        from strands_robots.tools.lerobot_calibrate import lerobot_calibrate

        res = await asyncio.to_thread(lerobot_calibrate, "list")
        return {"status": res.get("status"), "text": (res.get("content") or [{}])[0].get("text", "")}

    @app.get("/api/calibration/{name}")
    async def calibration_view(
        name: str,
        device_type: str | None = None,
        device_model: str | None = None,
    ) -> dict[str, Any]:
        """One calibration's per-motor detail."""
        from strands_robots.dashboard import calibration as calib
        from strands_robots.tools.lerobot_calibrate import lerobot_calibrate

        found = await asyncio.to_thread(
            calib.candidates,
            name,
            device_type=device_type,
            device_model=device_model,
        )
        if not found:
            raise HTTPException(
                404,
                {
                    "error": f"no calibration named {name!r}",
                    "hint": "GET /api/calibration lists every calibration on this machine",
                },
            )
        if len(found) > 1:
            raise HTTPException(
                409,
                {
                    "error": (f"{name!r} exists {len(found)} times - say which with ?device_type=&device_model="),
                    "candidates": found,
                },
            )
        target = found[0]
        res = await asyncio.to_thread(
            lambda: lerobot_calibrate(
                action="view",
                device_type=target["device_type"],
                device_model=target["device_model"],
                device_id=target["device_id"],
            )
        )
        content = res.get("content") or [{}]
        info: dict[str, Any] = {}
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("json"), dict):
                info = block["json"].get("calibration_info") or {}
        out: dict[str, Any] = {
            "status": res.get("status"),
            "text": content[0].get("text", "") if content else "",
        }
        out.update(calib.payload(info) if info else {k: target[k] for k in target})
        return out

    # ------------------------------------------------------------------
    # Calibration wizard — the interactive lerobot-calibrate flow, run as a
    # managed pty session so the operator never needs a terminal. Calibration
    # is torque-OFF (the dashboard commands no motion; the operator's hand does),
    # so the gate is the wizard's own confirm sheet, not the motion rail.
    # ------------------------------------------------------------------

    @app.post("/api/calibration/run")
    async def calibration_run_start(payload: dict[str, Any]) -> dict[str, Any]:
        from strands_robots.dashboard import calibration_run as cr

        port = str(payload.get("port") or "").strip()
        owner = app.state.devices.port_owner(port)
        if owner:
            raise HTTPException(
                409,
                {
                    "error": f"{port} is held by the running robot {owner!r} — two owners on one "
                    "servo bus is the 'Port is in use!' collision, and the wizard would measure "
                    "a bus that is mid-conversation",
                    "remedy": f"despawn {owner} first (its profile is remembered, respawn after)",
                },
            )
        try:
            run = cr.start(
                role=str(payload.get("role") or ""),
                model=str(payload.get("model") or ""),
                device_id=str(payload.get("device_id") or ""),
                port=port,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return run.status()

    @app.get("/api/calibration/run/{sid}")
    async def calibration_run_status(sid: str) -> dict[str, Any]:
        from strands_robots.dashboard import calibration_run as cr

        run = cr.get(sid)
        if run is None:
            raise HTTPException(
                404,
                f"no calibration session {sid!r} — it may have been "
                "superseded; start a new one from the devices drawer",
            )
        return run.status()

    @app.post("/api/calibration/run/{sid}/key")
    async def calibration_run_key(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
        from strands_robots.dashboard import calibration_run as cr

        run = cr.get(sid)
        if run is None:
            raise HTTPException(404, f"no calibration session {sid!r}")
        try:
            run.press(str(payload.get("key") or ""))
        except (ValueError, RuntimeError) as e:
            raise HTTPException(409, str(e)) from e
        return run.status()

    @app.post("/api/calibration/run/{sid}/cancel")
    async def calibration_run_cancel(sid: str) -> dict[str, Any]:
        from strands_robots.dashboard import calibration_run as cr

        run = cr.get(sid)
        if run is None:
            raise HTTPException(404, f"no calibration session {sid!r}")
        await asyncio.to_thread(run.cancel)
        return run.status()

    @app.get("/api/frame/{peer_id}/{cam}")
    async def frame(peer_id: str, cam: str) -> Response:
        f = app.state.bridge.latest_frame(peer_id, cam)
        if f is None:
            raise HTTPException(404, "no frame yet")
        if not f.get("jpeg"):
            # Serving raw pixels as image/jpeg renders as a black rectangle,
            # indistinguishable from a dead camera.
            raise HTTPException(415, f.get("error") or "frame is not displayable")
        return Response(content=f["jpeg"], media_type="image/jpeg")

    # ------------------------------------------------------------------
    # WebSockets
    # ------------------------------------------------------------------

    @app.websocket("/ws/mesh")
    async def ws_mesh(ws: WebSocket) -> None:
        await ws.accept()
        bridge: MeshBridge = app.state.bridge
        q = bridge.attach_queue()
        gone = asyncio.create_task(_client_gone(ws))
        try:
            await ws.send_text(json.dumps(bridge.snapshot()))
            while True:
                getter = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({getter, gone}, return_when=asyncio.FIRST_COMPLETED)
                if gone in done:
                    getter.cancel()
                    break
                await ws.send_text(json.dumps(getter.result()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            gone.cancel()
            bridge.detach_queue(q)

    @app.websocket("/ws/camera/{peer_id}/{cam}")
    async def ws_camera(ws: WebSocket, peer_id: str, cam: str) -> None:
        """Push binary JPEG frames for one tile. Sends only when a newer
        frame exists; paced at ~15 fps max so wifi phones stay happy."""
        await ws.accept()
        bridge: MeshBridge = app.state.bridge
        last_t: Any = None
        reported: str | None = None
        frames_sent = 0
        bytes_sent = 0
        started_at = time.monotonic()
        churn = getattr(ws.app.state, "camera_churn", _CAMERA_CHURN).note_open(
            viewer_identity(
                # A JWT is per-login, so its digest identifies the viewer without the token ever entering a
                # key or a log line.
                subject=(
                    hashlib.sha256(ws.query_params.get("token", "").encode()).hexdigest()[:12]
                    if ws.query_params.get("token")
                    else None
                ),
                host=ws.client.host if ws.client else None,
                peer_id=peer_id,
                cam=cam,
            )
        )
        cap = effective_cap(fps_cap(ws.query_params.get("max_fps")), churn.cap_fps)
        min_interval = None if cap is None else 1.0 / cap
        if churn.reason:
            # Say it on the tile, not only in the log: a silent throttle is indistinguishable from a slow
            # camera, and old bundles already render `camera_error` text - so even the tab that caused
            # this can explain itself.
            with contextlib.suppress(Exception):
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "camera_error",
                            "peer_id": peer_id,
                            "cam": cam,
                            "error": churn.reason,
                            "throttled": True,
                        }
                    )
                )
        last_sent_at: float | None = None
        gone = asyncio.create_task(_client_gone(ws))
        try:
            while not gone.done():
                if min_interval is not None and last_sent_at is not None:
                    waited = time.monotonic() - last_sent_at
                    if waited < min_interval:
                        # Skip WITHOUT consuming the frame: the newest one at the end of
                        # the wait is what the viewer wants, not this stale one.
                        await asyncio.sleep(min(min_interval - waited, 1 / 15))
                        continue
                f = bridge.latest_frame(peer_id, cam)
                if f is not None and f.get("t") != last_t:
                    last_t = f.get("t")
                    if f.get("jpeg"):
                        await ws.send_bytes(f["jpeg"])
                        frames_sent += 1
                        bytes_sent += len(f["jpeg"])
                        last_sent_at = time.monotonic()
                    elif f.get("error") and f["error"] != reported:
                        # One text frame per distinct problem: the tile can then
                        # say "raw frames, cannot decode" instead of going black.
                        reported = f["error"]
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "camera_error",
                                    "peer_id": peer_id,
                                    "cam": cam,
                                    "error": f["error"],
                                    "encoding": f.get("encoding"),
                                }
                            )
                        )
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            gone.cancel()
            # Rate-limited per peer/camera, with the suppressed count carried forward, so
            # a storm reads as a storm instead of drowning the log it would explain.
            log_now, suppressed = getattr(ws.app.state, "camera_close_log", _CAMERA_CLOSE_LOG).should_log(
                f"{peer_id}/{cam}"
            )
            if log_now:
                verdict = close_verdict(
                    frames_sent=frames_sent,
                    lifetime_s=time.monotonic() - started_at,
                    publishing=bridge.latest_frame(peer_id, cam) is not None,
                    bytes_sent=bytes_sent,
                )
                churn_note = (
                    ""
                    if not churn.throttled
                    else f" [server churn cap: {churn.opens_in_window} opens/min from this viewer]"
                )
                line = close_line(
                    peer_id=peer_id,
                    cam=cam,
                    verdict=verdict + cap_note(cap) + churn_note,
                    suppressed=suppressed,
                )
                (logger.info if frames_sent else logger.warning)(line)

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        """Fleet agent chat: streams token/reasoning/tool/done events."""
        import queue as _queue
        import threading as _threading

        from strands_robots.dashboard.agent_bridge import (
            _turn_lock,
            resume_interrupt_blocking,
            run_turn_blocking,
        )

        await ws.accept()
        try:
            while True:
                message = await ws.receive()
                turn, reply = parse_chat_frame(cast("dict[str, Any]", message))
                if reply is not None:
                    await ws.send_text(json.dumps(reply))
                    continue
                if turn is None:
                    continue
                if _turn_lock.locked():
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "notice",
                                "text": "another turn is running - yours is queued and will start when it finishes",
                            }
                        )
                    )
                q: _queue.Queue = _queue.Queue()
                cancel = _threading.Event()
                if isinstance(turn, dict):
                    # A yes/no on the pending motion confirm: resume the parked turn.
                    args: tuple[Any, ...] = (turn["interrupt_id"], turn["response"], q, cancel)
                    worker: Any = resume_interrupt_blocking
                else:
                    args = (turn, q, cancel)
                    worker = run_turn_blocking
                _threading.Thread(target=worker, args=args, daemon=True).start()
                try:
                    while True:
                        ev = await asyncio.to_thread(q.get)
                        if ev.get("type") == "__END__":
                            break
                        await ws.send_text(json.dumps(ev))
                except BaseException:
                    # Disconnect / cancellation / failed send: tell the worker
                    # to abandon the turn so it stops holding the turn lock.
                    cancel.set()
                    raise
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket) -> None:
        """Speech-to-speech fleet control (PCM16 <-> bidi agent)."""
        from strands_robots.dashboard.voice import run_voice_session

        await ws.accept()
        try:
            await run_voice_session(ws)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ Static PWA (built
    # frontend).

    if FRONTEND_DIST.exists():
        from fastapi.staticfiles import StaticFiles

        class _CachedStatic(StaticFiles):
            """StaticFiles that labels what may be cached (see static_cache_control)."""

            def file_response(self, *args: Any, **kwargs: Any) -> Response:
                resp = super().file_response(*args, **kwargs)
                resp.headers.setdefault(
                    "Cache-Control",
                    static_cache_control(getattr(resp, "path", "") or ""),
                )
                return resp

        app.mount("/assets", _CachedStatic(directory=FRONTEND_DIST / "assets"), name="assets")

        # Resolved once so the traversal check below has a canonical
        # ancestor to compare against - .resolve() on the user path each
        # request would race a symlink swap.
        _DIST_ROOT = FRONTEND_DIST.resolve()
        # Fallback SPA entry-point, computed once from the resolved root so
        # the ``FileResponse`` on the fallback branch never mentions user
        # input: it is a module-scope-visible constant to CodeQL.
        _INDEX_HTML = _DIST_ROOT / "index.html"
        # Whitelist regex for one SPA path segment. ``fullmatch`` against a
        # bounded character class is a barrier CodeQL's py/path-injection
        # sanitiser recognises; the segment-level split rejects ``..`` and
        # empty segments explicitly at the call site.
        _SAFE_SPA_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")

        def _admit_spa_path(raw: str) -> str | None:
            if not raw:
                return None
            for segment in raw.split("/"):
                if segment == "" or segment == "..":
                    return None
                if _SAFE_SPA_SEGMENT.fullmatch(segment) is None:
                    return None
            return raw

        @app.get("/{path:path}")
        async def spa(path: str) -> Response:
            # An unrouted /api path is a MISSING ENDPOINT, never a client-side route: answering it with
            # index.html made every typo and every renamed endpoint look like a 200 to the browser, so a
            # fetch() then died on "Unexpected token '<'" - or worse, a frontend feature-probe ("use the
            # real backend if it answers") concluded the endpoint existed and fed HTML-shaped junk into
            # its state.
            first = path.split("/", 1)[0]
            if first in ("api", "ws"):
                return JSONResponse(
                    {"error": "not found", "detail": f"no endpoint at /{path}"},
                    status_code=404,
                )

            # The URL path is user-supplied and reaches the filesystem here.
            # Without confinement, ``GET /../../etc/passwd`` resolves outside
            # FRONTEND_DIST and FileResponse serves it - a real filesystem
            # read on a path FastAPI happily forwards.
            #
            # Three layers of confinement, in order:
            #
            # 1. Whitelist regex on each segment via ``_admit_spa_path``.
            #    ``fullmatch`` against a bounded character class is a barrier
            #    CodeQL's py/path-injection sanitiser recognises - the sink
            #    below sees a value that has been through a regex clean.
            #
            # 2. Segment-level rejection of ``..`` and empty segments (from
            #    leading or repeated slashes) inside ``_admit_spa_path``.
            #
            # 3. Post-resolve ``is_relative_to`` on the resolved candidate
            #    against the resolved dist root - belt-and-braces against a
            #    symlink inside the tree that points outside.
            safe_path = _admit_spa_path(path)
            if safe_path is not None:
                candidate = (FRONTEND_DIST / safe_path).resolve()
                # ``is_relative_to`` returns False for a path that resolves
                # outside _DIST_ROOT even via a symlink - the last-line guard
                # against a symlink inside the tree that points elsewhere.
                if candidate.is_relative_to(_DIST_ROOT) and candidate.is_file():
                    return FileResponse(
                        candidate,
                        headers={"Cache-Control": static_cache_control(safe_path)},
                    )
            # An SPA route falls back to the entry point, so it is labelled
            # like one. ``_INDEX_HTML`` is a closed-over constant computed at
            # mount time from ``_DIST_ROOT``; no user input flows into it.
            return FileResponse(
                _INDEX_HTML,
                headers={"Cache-Control": static_cache_control("index.html")},
            )
    else:

        @app.get("/")
        async def no_frontend() -> dict[str, str]:
            return {
                "message": "frontend not built - run: cd strands_robots/dashboard/frontend && npm install && npm run build",
                "api": "/api/health",
            }

    return app
