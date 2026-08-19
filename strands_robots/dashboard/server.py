"""FastAPI app for the Strands Robots Dashboard.

Endpoints:
    GET  /api/health                     liveness + mesh status (always public)
    GET  /api/fleet                      current fleet snapshot
    GET  /api/robots/registry            registered robot names (spawnable)
    GET  /api/policies                   full provider catalog (run-form schema)
    POST /api/policies/validate          pre-flight a provider config
    POST /api/robots/{peer}/task         start a task on a peer
    POST /api/robots/{peer}/stop         stop the running task on a peer
    POST /api/robots/{peer}/teleop/publish  publish a leader's own joints
    POST /api/safety/estop               fleet-wide emergency stop
    GET  /api/config                     agent / voice / mesh / env config
    POST /api/config                     apply config (hot where possible)
    GET  /api/mesh/config                mesh endpoints + wire-security posture
    POST /api/mesh/config                re-point the mesh (optional restart)
    GET  /api/agent/status               fleet-agent readiness
    POST /api/agent/reset                rebuild the agent / clear history
    GET  /api/activity                   recent fleet commands + safety events
    GET  /api/frame/{peer}/{cam}         latest JPEG frame (poll)
    WS   /ws/mesh                        live event stream
    WS   /ws/camera/{peer}/{cam}         binary JPEG stream for one camera tile
    /                                    static PWA (frontend/dist)

Auth: when ``security.auth_token`` is configured (settings or
``DASHBOARD_AUTH_TOKEN``) every /api and /ws request must present it. With no
token configured the server stays open - the LAN-dev posture the dashboard was
built with - and the UI shows an explicit "unauthenticated" warning rather than
pretending otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from strands_robots.dashboard import arm_roles, config_api, consent, settings
from strands_robots.dashboard.teleop_health import published_frames, teleop_health
from strands_robots.dashboard.device_manager import DeviceManager
from strands_robots.dashboard.mesh_bridge import MeshBridge, stop_outcome

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

#: Reachable without a token: liveness (so a client can discover that auth is
#: required at all), the WebAuthn ceremony endpoints (you cannot log in from
#: behind a wall that requires being logged in), and the static shell (which
#: renders the login prompt). /api/auth/register/* gates ITSELF: once a
#: passkey exists, enrolling another requires a valid session.
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
    """Bearer-token gate for /api and /ws, as raw ASGI.

    Raw ASGI rather than ``BaseHTTPMiddleware`` because WebSocket scopes never
    reach an HTTP middleware - and /ws/mesh, /ws/chat and /ws/voice are exactly
    the endpoints that drive motors and spend money.

    Accepted credentials, in order: the static ``security.auth_token``, then a
    WebAuthn session JWT minted by ``dashboard.auth``. With NEITHER configured
    the open posture is LOCAL-ONLY: loopback clients pass (the LAN-dev
    workflow), anything else is refused - an unauthenticated dashboard
    commands real motors, so "auth disabled" must never mean "open to the
    network".
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    @staticmethod
    def _client_is_local(scope: dict[str, Any]) -> bool:
        # A reverse proxy (cloudflared tunnel) connects FROM loopback on behalf
        # of the whole internet. Any forwarding header means the ORIGINAL
        # client is remote, so "local" must be false no matter the socket peer.
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
            return auth[7:].strip()
        if headers.get("x-dashboard-token"):
            return headers["x-dashboard-token"].strip()
        # Browsers cannot set headers on a WebSocket handshake, so the query
        # string is the only channel there.
        query = parse_qs(scope.get("query_string", b"").decode())
        return (query.get("token") or [""])[0].strip()

    @staticmethod
    def _cross_origin_refused(scope: dict[str, Any]) -> bool:
        """True when a BROWSER cross-origin request must be refused.

        CORS only protects responses; the SIDE EFFECT of a mutating request
        fires before the browser hides the reply, and a no-header POST (e.g.
        an e-stop, which needs no body) is a "simple request" the browser
        sends without any preflight. So the guard itself refuses writes and
        websocket handshakes whose Origin disagrees with the request host and
        is not explicitly allow-listed. Non-browser clients (curl, scripts,
        the spawn watcher) send no Origin header and are untouched.
        """
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        origin = headers.get("origin", "").strip()
        if not origin:
            return False
        host = headers.get("host", "").strip().lower()
        netloc = urlsplit(origin).netloc.strip().lower()
        if netloc and netloc == host:
            return False  # same-origin
        # A WILDCARD IS NOT A WRITE PERMIT. "*" is a reasonable answer to
        # "who may READ this API", and it is what older installs persisted into
        # settings.json - but honouring it here would let any tab the operator
        # happens to have open POST /api/robots/{peer}/task and move the arms
        # (Q20). Mutations and websockets need an origin named EXPLICITLY.
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
                await self.app(scope, receive, send)
                return
        if scope["type"] == "websocket":
            await receive()  # consume websocket.connect before rejecting
            await send({"type": "websocket.close", "code": 1008})
            return
        response = JSONResponse({"detail": "unauthorized"}, status_code=401)
        await response(scope, receive, send)


CHAT_MAX_FRAME_BYTES = 32 * 1024  # a generous chat turn; 2 MB frames ran real model turns


def parse_chat_frame(message: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """One /ws/chat frame -> ``(prompt, reply)``.

    Exactly one of the two is non-None, or both are None (nothing to do).
    A protocol error is a typed ``error`` REPLY, never a prompt: junk frames
    used to be promoted to prompts and billed as model turns (Q18), and a
    binary frame or non-string ``text`` killed the socket outright (Q17).
    Raises WebSocketDisconnect for a disconnect message.
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
    text = msg.get("text")
    if text is None:
        return None, None
    if not isinstance(text, str):
        return None, {"type": "error", "error": f"text must be a string, got {type(text).__name__}"}
    prompt = text.strip()
    return (prompt or None), None


def _audit_autospawn(bridge: Any, did: dict[str, Any] | None) -> None:
    """Land the auto-spawn watcher's poll results in the activity trail.

    The watcher spawns robots with NOBODY at the keyboard - the one actor
    whose actions most need an audit trail. poll() already reports what it
    did; without this, a board plugged in while the operator is away becomes
    a moving arm with no recorded cause.
    """
    if not did:
        return
    for peer_id in did.get("spawned") or []:
        bridge.record_activity(
            "api", "spawn", target=peer_id,
            detail="USB auto-spawn (board plugged in)", ok=True,
        )
    for peer_id in did.get("despawned") or []:
        bridge.record_activity(
            "api", "despawn", target=peer_id,
            detail="USB auto-spawn (board unplugged)", ok=True,
        )


def create_app(bridge: MeshBridge | None = None) -> FastAPI:
    app = FastAPI(title="strands-robots dashboard")
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
    app.state.mesh_online = False
    app.state.devices = DeviceManager()
    # /api/record - teleop episode recording (record screen). The controller
    # parks the arms' fleet peers around a session; see record_api.py.
    from strands_robots.dashboard import record_api

    app.state.record = record_api.RecordController(app.state.devices)
    # Late-bound on purpose: capturing the bound method here would pin the
    # router to THIS bridge instance forever (tests swap it, restart_mesh may).
    app.include_router(record_api.build_router(
        app.state.record,
        on_activity=lambda *a, **k: app.state.bridge.record_activity(*a, **k),
    ))
    # A peer with a LIVE managed local process is never aged out of the fleet
    # snapshot, even if its state stream goes quiet.
    app.state.bridge.protected_peer_ids = lambda: {
        pid for pid, m in list(app.state.devices.robots.items()) if m.alive()
    }
    # The measured leader/follower role travels with the fleet snapshot (both
    # rails), so a card can say which arm it IS instead of leaving the operator to
    # infer it from a name. Cached server-side; /api/fleet is polled ~1Hz.
    app.state.bridge.peer_annotations = app.state.devices.roles_by_peer

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        app.state.mesh_online = await asyncio.to_thread(app.state.bridge.start, loop)
        # Hand the mesh gateway to the fleet agent (chat + voice).
        from strands_robots.dashboard import agent_bridge as _ab

        _ab.set_bridge(app.state.bridge)

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

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mesh_online": app.state.mesh_online,
            "dashboard_peer_id": app.state.bridge.peer_id,
            "peers": len(app.state.bridge.peers),
            # How much /ws/mesh fan-out the coalescer avoided. Reported rather
            # than asserted: a saving nobody can measure is a claim, and this is
            # the number the perf lens should be able to check on a live fleet.
            "mesh_coalesce": app.state.bridge.coalesce_stats(),
            "t": time.time(),
        }

    @app.get("/api/fleet")
    async def fleet() -> dict[str, Any]:
        # Roles ride along inside snapshot() (bridge.peer_annotations), so this
        # route and the WS stream cannot disagree about which arm is the leader.
        return app.state.bridge.snapshot()

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

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        out = mod.status(request)
        out["authenticated"] = mod.session_is_valid(_session_presented(request))
        return out

    @app.post("/api/auth/register/begin")
    async def auth_register_begin(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json() if await request.body() else {}
        if mod.has_credentials() and not mod.session_is_valid(_session_presented(request)):
            raise HTTPException(401, "enrolling another passkey requires a signed-in session")
        return mod.begin_registration(
            request,
            label=str(body.get("label") or "passkey")[:64],
            bootstrap=str(body.get("bootstrap") or ""),
        )

    @app.post("/api/auth/register/finish")
    async def auth_register_finish(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json()
        first_time = not mod.has_credentials()
        if not first_time and not mod.session_is_valid(_session_presented(request)):
            raise HTTPException(401, "enrolling another passkey requires a signed-in session")
        return mod.finish_registration(request, body.get("challenge_id", ""), body.get("credential") or {})

    @app.post("/api/auth/login/begin")
    async def auth_login_begin(request: Request) -> dict[str, Any]:
        return _require_auth_module().begin_authentication(request)

    @app.post("/api/auth/login/finish")
    async def auth_login_finish(request: Request) -> dict[str, Any]:
        mod = _require_auth_module()
        body = await request.json()
        return mod.finish_authentication(request, body.get("challenge_id", ""), body.get("credential") or {})

    @app.get("/api/auth/credentials")
    async def auth_credentials() -> dict[str, Any]:
        # reached only with a valid session (guarded path, not in PUBLIC_PATHS)
        return {"credentials": _require_auth_module().list_credentials()}

    @app.delete("/api/auth/credentials/{cred_id}")
    async def auth_credential_delete(cred_id: str) -> dict[str, Any]:
        return _require_auth_module().delete_credential(cred_id)

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
        """Full provider catalog - the schema the run form is generated from.

        ``requires``/``config_keys``/``defaults`` come straight from
        ``registry/policies.json``, so a provider that needs a port or a
        checkpoint says so instead of failing mid-run.
        """
        catalog = await asyncio.to_thread(config_api._policy_catalog)
        if not catalog:
            raise HTTPException(500, "policy registry unavailable")
        return {"providers": catalog, "names": [p["name"] for p in catalog]}

    def require_peer(peer_id: str) -> None:
        """404 for a peer that was never in the fleet, before spending the RPC.

        Every /api/robots/{peer}/* route used to send the command and wait out its
        whole timeout -- 10s for stop, up to duration+10 for a task -- then answer
        200 with state "no_answer". That is the SAME word a real robot that went
        quiet produces, so a typo and a wedged arm were indistinguishable, on the
        stop path of all places. A known-but-stale peer is still addressed: "it
        went quiet, try stopping it anyway" is a real thing to want.
        """
        from strands_robots.dashboard.mesh_bridge import peer_is_known

        bridge = app.state.bridge
        managed = getattr(getattr(app.state, "devices", None), "robots", None) or {}
        if peer_is_known(peer_id, getattr(bridge, "peers", None) or {}, managed):
            return
        known = sorted(set(getattr(bridge, "peers", None) or {}) | set(managed))
        raise HTTPException(404, {
            "error": f"no peer {peer_id!r} in the fleet",
            "hint": "GET /api/fleet lists the peers that can be commanded",
            "known_peers": known,
        })

    @app.get("/api/robots/{peer_id}/teleop")
    async def teleop_status(peer_id: str) -> dict[str, Any]:
        """Live teleop health for one peer: publisher/receiver rates, drops,
        slew rejections - the counters InputPublisher/InputReceiver already
        keep. Works on hardware AND sim peers (TeleopMixin lift)."""
        require_peer(peer_id)
        result = await app.state.bridge.send_cmd_async(peer_id, {"action": "teleop_status"}, timeout=10.0)
        # The counters alone lied: a follower refusing EVERY frame reported
        # running:true, and the reason existed only in its child log. health turns
        # the counters (+ that log, when the peer is ours) into a sentence, and a
        # refusal we can continue from arrives with its consent request attached.
        inner = result.get("result") if isinstance(result, dict) else None
        log_tail = None
        managed = app.state.devices.robots.get(peer_id)
        if managed is not None:
            log_tail = list(getattr(managed, "logs", []) or [])[-40:]
        health = teleop_health(inner if inner is not None else result, log_tail)
        # "Nothing is arriving" is not yet an answer: it could be a leader that
        # never started, or two peers that are not meeting. Ask the named leader
        # ONLY in that case - one extra round trip, and only when it decides
        # which end of the problem the operator should look at.
        silent = {k: v for k, v in health.get("receivers", {}).items() if v.get("state") == "silent"}
        if silent:
            counted: dict[str, int] = {}
            for key in silent:
                source, _, device = key.partition("/")
                if not source or source == peer_id:
                    continue
                try:
                    src = await app.state.bridge.send_cmd_async(
                        source, {"action": "teleop_status"}, timeout=10.0)
                except Exception as exc:  # noqa: BLE001 - a quiet leader is data too
                    logger.debug("could not ask leader %s about its publisher: %r", source, exc)
                    continue
                frames = published_frames(
                    src.get("result") if isinstance(src, dict) else src, device or "leader")
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
        """Make a peer a teleop SOURCE from its own joints (U3, leader half).

        Until this existed the chain was half-built: /teleop/receive could point
        a follower at a leader stream, but nothing on the mesh could make that
        stream exist, so the follower waited out its subscribe budget and
        answered with a shrug.

        Read-only on the arm named here - it publishes what it measures and moves
        nothing. The mover is whoever is pointed at this stream.
        """
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
        """Collect a policy-driven dataset in a one-shot mesh sim.

        run_policy drives exactly n_episodes rollouts with per-episode
        parquet boundaries and reports parquet-truth counts. The dataset
        lands under dataset_root, where /api/training/datasets discovers it
        - closing the record -> train -> deploy loop entirely in the UI.
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
        return result

    @app.post("/api/replay")
    async def replay_episode(body: dict[str, Any]) -> dict[str, Any]:
        """Replay a recorded LeRobotDataset episode in a one-shot mesh sim.

        The replay peer appears in the fleet grid with live cameras while
        the recorded actions drive real MuJoCo physics; it exits when the
        episode ends. Datasets come from /api/training/datasets.
        """
        repo_id = (body.get("repo_id") or "").strip()
        if not repo_id:
            raise HTTPException(422, "repo_id required")
        # Values go to the validator UNCOERCED: int("banana") here was a 500
        # wearing a stack trace, and int() would also quietly turn 5.9 into 5
        # before anything could refuse it. validate_replay judges the actual
        # request; a bad one is a 422 naming what to change (Q5).
        from strands_robots.dashboard.device_manager import validate_replay

        bad = validate_replay(
            repo_id, body.get("episode", 0), body.get("root"), body.get("speed", 1.0)
        )
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
        return result

    @app.get("/api/training/trainers")
    async def training_trainers() -> dict[str, Any]:
        from strands_robots.dashboard import training

        return {"trainers": await asyncio.to_thread(training.list_trainers)}

    @app.get("/api/training/datasets")
    async def training_datasets(q: str = "") -> dict[str, Any]:
        """Local LeRobotDataset roots for the submit form's dataset picker."""
        from strands_robots.dashboard import training

        return {"datasets": await asyncio.to_thread(training.local_datasets, q)}

    @app.get("/api/training/jobs")
    async def training_jobs() -> dict[str, Any]:
        from strands_robots.dashboard import training

        return {"jobs": await asyncio.to_thread(training.jobs)}

    @app.post("/api/training/validate")
    async def training_validate(body: dict[str, Any]) -> dict[str, Any]:
        from strands_robots.dashboard import training

        return await asyncio.to_thread(training.validate, body)

    @app.post("/api/training/submit")
    async def training_submit(body: dict[str, Any]) -> dict[str, Any]:
        """Launch a training job (train_policy validates the spec first)."""
        from strands_robots.dashboard import training

        result = await asyncio.to_thread(training.submit, body)
        job = (result.get("data") or {}).get("job_id") if isinstance(result, dict) else None
        app.state.bridge.record_activity(
            "training", "submit", target=str(job or body.get("provider", "?")),
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
        """Type-ahead checkpoint search for the run form.

        Merges the local HF cache (instant, marked ``local``) with a Hub
        search of public LeRobot checkpoints ranked by downloads. Each row
        is ready to drop into ``pretrained_name_or_path`` and carries a
        best-effort ``policy_type`` prefill for lerobot_async's required
        field.
        """
        from strands_robots.dashboard import checkpoints

        return await asyncio.to_thread(checkpoints.search, q, checkpoints.clamp_limit(limit))

    @app.get("/api/checkpoints/families")
    async def checkpoint_families() -> dict[str, Any]:
        """policy_type values the lerobot family accepts (type dropdown)."""
        from strands_robots.dashboard import checkpoints

        return {"families": await asyncio.to_thread(checkpoints.policy_families)}

    @app.post("/api/policies/validate")
    async def validate_policy(body: dict[str, Any]) -> dict[str, Any]:
        """Dry-run a provider config without touching a robot.

        Answers "will this run?" locally - missing dep, unknown provider,
        unreachable inference server, untrusted remote code - instead of making
        the operator read it out of a peer's 30 s timeout.
        """
        provider = (body.get("policy_provider") or "").strip()
        if not provider:
            raise HTTPException(422, "policy_provider required")
        config = body.get("policy_config") or {}
        if not isinstance(config, dict):
            raise HTTPException(422, "policy_config must be an object")

        # Preflight's most useful check compares the model's declared image
        # inputs against the observation keys it will actually receive, so pass
        # the target peer's real joints + cameras when we know them.
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
        if not observation_keys:
            result["note"] = (
                "no live observation keys for that peer - camera/joint routing "
                "was not checked"
            )
        return result

    @app.post("/api/robots/{peer_id}/task")
    async def start_task(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        require_peer(peer_id)
        instruction = (body.get("instruction") or "").strip()
        if not instruction:
            raise HTTPException(422, "instruction required")
        duration = float(body.get("duration", 30.0))
        cmd = {
            # Sim peers accept "start"; hardware peers accept both. "execute"
            # is kept as an explicit opt-in for callers that need it.
            "action": body.get("action") or "start",
            "instruction": instruction,
            "policy_provider": body.get("policy_provider", "mock"),
            "duration": duration,
        }
        # Only wire-settable keys: validate_command() builds its output from a
        # strict per-action allowlist and *silently drops* everything else, so
        # forwarding e.g. a policy_config dict would look accepted and arrive
        # empty. config_api.WIRE_CMD_KEYS is that allowlist.
        for opt in config_api.WIRE_CMD_KEYS:
            if body.get(opt) is not None:
                cmd[opt] = body[opt]
        # Child sim peers can't execute themselves:
        # route "<parent>__<robot>" to the parent with robot_name here, so
        # the card ▶ button and every API caller get the fix - not just the
        # agent's fleet tool.
        from strands_robots.dashboard.mesh_bridge import route_task_target

        target, cmd = route_task_target(peer_id, cmd)
        # The peer only answers when the policy *finishes*, so a timeout below
        # duration reports a bare "timeout" on a task that is running fine.
        timeout_s = max(float(body.get("timeout", 60.0)), duration + 10.0)
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
        """Fleet-wide stop: broadcast {action: stop} to every live peer.

        Note: the signed strands/safety/estop envelope requires a Mesh
        instance with a robot, so the dashboard sends per-peer stop commands
        instead. The full lockout-engaging e-stop lands once the dashboard
        embeds a local sim peer or lifts envelope signing into the bridge.

        Only *live* peers are addressed, and every peer is classified
        stopped / not_stopped / no_answer. A stale peer counted as "stopped"
        is exactly the lie an e-stop must never tell.
        """
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
            "estop", "stop_all", target="fleet",
            detail=f"{counts['stopped']}/{len(peers)} confirmed stopped",
            ok=counts["stopped"] == len(peers) and bool(peers),
        )
        # A6: fire the SIGNED safety rail too - the envelope engages the
        # fleet-wide LOCKOUT on every listening peer (they refuse all further
        # commands until a proofed resume), which per-peer stop commands
        # cannot do. Signed rail failure must not degrade the broadcast-stop
        # above; both fire, results are reported side by side.
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
        """Clear the fleet e-stop lockout with the operator override code.

        The code is verified locally (brute-force throttled) and the
        HMAC-proofed resume envelope is published for every peer to
        re-verify independently. The code itself never crosses the wire.
        """
        code = (body.get("override_code") or "").strip()
        if not code:
            raise HTTPException(422, "override_code required")
        result = await asyncio.to_thread(app.state.bridge.signed_resume, code)
        bridge2: MeshBridge = app.state.bridge
        bridge2.record_activity(
            "resume", "safety_resume", target="fleet",
            detail=result.get("status", result.get("error", "?")),
            ok=result.get("status") == "ok",
        )
        return result

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
        """What this machine currently grants, and what can be asked for (U18)."""
        allow = [e.strip() for e in os.environ.get("STRANDS_MESH_HF_REPO_ALLOW", "").split(",") if e.strip()]
        return {
            "kinds": list(consent.KINDS),
            "trust_remote_code": os.environ.get("STRANDS_TRUST_REMOTE_CODE", "").strip().lower()
            in ("1", "true", "yes"),
            "hf_repo_allow": allow,
            "env_file": str(config_api.ENV_FILE),
        }

    @app.post("/api/consent")
    async def post_consent(body: dict[str, Any]) -> dict[str, Any]:
        """Approve one refusal, by kind + subject — nothing else is settable.

        The browser never sends the variable or the value: the request is
        REBUILT here from ``kind``/``subject`` so an approval can only ever
        widen the exact guard the SDK named. A grant reaches this process'
        environment (so the next spawned child inherits it) and .env (so it
        survives a restart), and the answer says plainly whether the peer that
        was refused needs a respawn to see it.
        """
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

            _mesh_security.hf_repo_allowlist()
        except Exception:  # noqa: BLE001 - a cache warm-up must not fail a grant
            logger.debug("consent: allowlist warm-up failed", exc_info=True)
        app.state.bridge.record_activity(
            "api", "consent", target=request.scope,
            detail=f"approved: {', '.join(request.grants)}", ok=True,
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
        """Take one grant back — the other half of a promise the dialog makes.

        Narrow in the same way the grant was: revoking one repository leaves the
        rest of the allowlist untouched. Already-running children keep the
        permission they were started with, and the answer says so instead of
        implying the fleet was locked down retroactively.
        """
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
            "api", "consent", target=request.scope, detail="revoked", ok=True,
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
        """Re-open the mesh session against the current settings.

        Guarded because the session is a ref-counted module singleton: locally
        spawned robots hold their own references and would keep talking to the
        old endpoints, so re-pointing under them orphans them from the
        dashboard's view of the fleet.
        """
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
        return app.state.bridge.mesh_info()

    @app.post("/api/mesh/config")
    async def post_mesh_config(body: dict[str, Any]) -> dict[str, Any]:
        """Persist mesh endpoints and (by default) re-point the session."""
        body = body or {}
        mesh = {k: v for k, v in body.items()
                if k in ("connect", "listen", "port", "backend", "camera_hz", "policy_type_allow")}
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
        """peer_id -> camera names the mesh has actually seen frames for.

        The evidence for "in use": a camera in a child's config that never
        delivered a frame is assigned, not streaming, and only the frames can
        tell the two apart.
        """
        snapshot = app.state.bridge.snapshot()
        return {
            peer_id: list((entry.get("cameras") or {}).keys())
            for peer_id, entry in (snapshot.get("peers") or {}).items()
        }

    @app.get("/api/devices")
    async def devices(refresh: bool = False) -> dict[str, Any]:
        """Local USB serial ports (servo buses) + cameras + managed robots.

        Camera probe results are cached ~30s and indices owned by running
        robots are never re-opened; ``?refresh=1`` forces a
        fresh probe of the unclaimed indices.
        """
        # The mesh's frame bookkeeping is the evidence for "in use": a camera
        # in a child's config that never delivered a frame is assigned, not
        # streaming, and the difference is what the operator has to act on.
        return await asyncio.to_thread(app.state.devices.devices, refresh, _live_camera_names())

    @app.get("/api/devices/camera/{index}/preview")
    async def camera_preview(index: int) -> Response:
        """One JPEG frame from an unclaimed camera index.

        The authoritative "which camera is index N" answer - device names are
        listed in a different order than OpenCV indices on macOS, so the
        picture is the identity. 409 when the index is streaming for a
        running robot (watch that robot's card instead), 503 when the camera
        will not produce a frame (unplugged, or another app holds it).
        """
        try:
            jpeg = await asyncio.to_thread(
                app.state.devices.preview_frame, index, _live_camera_names(),
            )
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
            raise HTTPException(503, str(e)) from e
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/devices/camera/{index}/modes")
    async def camera_modes(index: int) -> dict[str, Any]:
        """Verified fps/resolution modes for an unclaimed camera (U19).

        Each candidate mode is set and read back on the real device; only
        combos the camera AGREED to (plus its native mode) come back, so the
        reconfigure sheet's selects never offer a fantasy the driver would
        silently ignore. 409 while the index streams for a running robot,
        503 when the camera will not open.
        """
        try:
            return await asyncio.to_thread(
                app.state.devices.probe_modes, index, _live_camera_names(),
            )
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
            raise HTTPException(503, str(e)) from e

    @app.get("/api/devices/arm-role")
    async def arm_role(port: str, model: str = "sts3215") -> dict[str, Any]:
        """Which role an arm actually IS, read off its servo bus (U2).

        An SO-100/SO-101 follower runs a 12V bus, a leader 7.4V, and every
        Feetech servo reports its own supply on the read-only Present_Voltage
        register - so the role is measurable instead of inherited from whatever
        name a profile was given. The operator's report was that the dashboard
        has the two arms the wrong way round; a label cannot answer that, a
        measurement can.

        Register READS only: this cannot move an arm. 409 while a live child
        holds the port - a servo bus has exactly one owner, and that child is it.
        """
        try:
            # Measures AND remembers (keyed by USB serial, never by /dev name -
            # the OS reassigns those). A measurement that lives only in one HTTP
            # response does not fix the label the operator sees next session.
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
        # An unspawnable mode or an unknown robot is a bad REQUEST, answered
        # before any process exists. It used to reach Popen: the child raised, the
        # route had already reported a pid, and mode="quantum" quietly produced a
        # sim peer wearing "quantum" as its label (the spawner branches on
        # mode == "real" and sims everything else, so "Real" did it too).
        from strands_robots.dashboard.device_manager import validate_spawn

        checked = await asyncio.to_thread(validate_spawn, robot_name, body.get("mode", "sim"))
        if isinstance(checked, dict):
            app.state.bridge.record_activity(
                "api", "spawn", target=str(robot_name),
                detail=f"refused: {checked['error']}", ok=False,
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
        # A pid is not a running robot. Wait out the window in which a
        # misconfigured child dies (wrong camera config, port held by another
        # process, policy not installed) so the answer is what happened, not
        # what was attempted. Returns early the moment the mesh sees the peer.
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
                # A refusal the operator can answer travels as needs_consent, so
                # the UI offers "Approve & retry" instead of a wall of prose
                # whose only remedy is a shell (U18).
                consent.attach_consent(
                    result, result["error"], "\n".join(outcome.get("log_tail") or [])
                )

        # Lifecycle lands in the audit trail: "who started this peer" is as
        # unanswerable as "who moved that arm" without it - the auto-spawn
        # watcher and the UI use this same route.
        app.state.bridge.record_activity(
            "api", "spawn", target=result.get("peer_id") or robot_name,
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
        return result

    @app.get("/api/devices/profiles")
    async def device_profiles() -> dict[str, Any]:
        """Remembered USB device profiles, keyed by board serial number.

        A profile is written whenever a real (serial-port) robot is spawned
        successfully, and it is what the auto-spawn watcher replays when that
        exact board is plugged back in.
        """
        return {
            "profiles": app.state.devices.profiles.all(),
            "path": app.state.devices.profiles.path,
            "autospawn": getattr(app.state, "autospawn_task", None) is not None,
        }

    @app.post("/api/devices/despawn")
    async def despawn(body: dict[str, Any]) -> dict[str, Any]:
        peer_id = body.get("peer_id")
        if not peer_id:
            raise HTTPException(422, "peer_id required")
        result = await asyncio.to_thread(app.state.devices.despawn, peer_id)
        app.state.bridge.record_activity(
            "api", "despawn", target=peer_id, ok="error" not in result,
        )
        return result

    @app.post("/api/devices/{peer_id}/cameras")
    async def reconfigure_cameras(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Respawn a managed peer with a new camera config (U19 v1).

        Cameras are taken only at spawn, so attach/detach/fps/resolution is
        honestly a respawn - one named atomic operation. The peer's streams
        drop for the settle window; the UI's confirm dialog owns that consent.
        ``cameras: null`` detaches everything; each entry is lerobot-shaped
        ({name: {index_or_path, fps?, width?, height?}}). Invalid configs are
        422 BEFORE the running peer is touched - a refusal must never cost the
        operator the process they already had.
        """
        if "cameras" not in body:
            raise HTTPException(422, "cameras required (a mapping, or null to detach all)")
        from strands_robots.dashboard.device_manager import validate_cameras

        bad = validate_cameras(body.get("cameras"))
        if bad:
            raise HTTPException(422, bad)
        result = await asyncio.to_thread(
            app.state.devices.reconfigure_cameras, peer_id, body.get("cameras")
        )
        if "error" in result and not result.get("reconfigured"):
            app.state.bridge.record_activity(
                "api", "cameras", target=peer_id,
                detail=f"refused: {result['error']}", ok=False,
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
            "api", "cameras", target=peer_id,
            detail=f"respawned with {len(body.get('cameras') or {})} camera(s)"
            + (f" -> {result['error']}" if result.get("error") else ""),
            ok="error" not in result,
        )
        return result

    @app.get("/api/devices/logs/{peer_id}")
    async def device_logs(peer_id: str) -> dict[str, Any]:
        """Child-process output for one managed robot (ring buffer)."""
        out = app.state.devices.logs(peer_id)
        if "error" in out:
            # Q23, same errors-as-200 family as Q3: res.ok must not be true for a
            # peer that does not exist. Only LOCALLY SPAWNED robots have logs at
            # all, so the message says which ids qualify rather than implying the
            # peer is unknown to the whole fleet.
            managed = sorted(getattr(app.state.devices, "robots", None) or {})
            raise HTTPException(404, {
                "error": out["error"],
                "hint": "only locally spawned robots keep a log ring buffer",
                "managed_peers": managed,
            })
        return out

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
        """One calibration's per-motor detail.

        This route used to call the tool POSITIONALLY, so ``name`` landed in
        ``device_type`` and it answered "view action requires: device_type,
        device_model, and device_id" for every input -- the drawer rendered that
        sentence as if it were data.

        A name is not an identity: ``leader_arm`` exists under three models on
        this machine. With no ``device_model`` the route resolves it, and when the
        name is ambiguous it answers **409 with the candidates** rather than
        picking one -- showing ``so_follower``'s numbers to someone who meant
        ``so101_follower`` is worse than a question. Returns structured
        ``motors`` (the tool has always carried them; the UI was parsing
        markdown) alongside the original ``text``.
        """
        from strands_robots.dashboard import calibration as calib
        from strands_robots.tools.lerobot_calibrate import lerobot_calibrate

        found = await asyncio.to_thread(
            calib.candidates, name, device_type=device_type, device_model=device_model,
        )
        if not found:
            raise HTTPException(404, {
                "error": f"no calibration named {name!r}",
                "hint": "GET /api/calibration lists every calibration on this machine",
            })
        if len(found) > 1:
            raise HTTPException(409, {
                "error": (
                    f"{name!r} exists {len(found)} times - say which with "
                    "?device_type=&device_model="
                ),
                "candidates": found,
            })
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
        try:
            await ws.send_text(json.dumps(bridge.snapshot()))
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bridge.detach_queue(q)

    @app.websocket("/ws/camera/{peer_id}/{cam}")
    async def ws_camera(ws: WebSocket, peer_id: str, cam: str) -> None:
        """Push binary JPEG frames for one tile. Sends only when a newer
        frame exists; paced at ~15 fps max so wifi phones stay happy."""
        await ws.accept()
        bridge: MeshBridge = app.state.bridge
        last_t: Any = None
        reported: str | None = None
        try:
            while True:
                f = bridge.latest_frame(peer_id, cam)
                if f is not None and f.get("t") != last_t:
                    last_t = f.get("t")
                    if f.get("jpeg"):
                        await ws.send_bytes(f["jpeg"])
                    elif f.get("error") and f["error"] != reported:
                        # One text frame per distinct problem: the tile can then
                        # say "raw frames, cannot decode" instead of going black.
                        reported = f["error"]
                        await ws.send_text(json.dumps({
                            "type": "camera_error", "peer_id": peer_id, "cam": cam,
                            "error": f["error"], "encoding": f.get("encoding"),
                        }))
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        """Fleet agent chat: streams token/reasoning/tool/done events."""
        import queue as _queue
        import threading as _threading

        from strands_robots.dashboard.agent_bridge import _turn_lock, run_turn_blocking

        await ws.accept()
        try:
            while True:
                message = await ws.receive()
                prompt, reply = parse_chat_frame(message)
                if reply is not None:
                    await ws.send_text(json.dumps(reply))
                    continue
                if prompt is None:
                    continue
                # Q19: turns are serialized by _turn_lock (chat + voice). The
                # second client used to stare at a dead chat box for the whole
                # first turn (measured 21.8s) - say so, in a 'notice' the
                # frontend already renders.
                if _turn_lock.locked():
                    await ws.send_text(json.dumps({
                        "type": "notice",
                        "text": "another turn is running - yours is queued and will start when it finishes",
                    }))
                q: _queue.Queue = _queue.Queue()
                cancel = _threading.Event()
                _threading.Thread(
                    target=run_turn_blocking, args=(prompt, q, cancel), daemon=True,
                ).start()
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

    # ------------------------------------------------------------------
    # Static PWA (built frontend). Fallback to index.html for SPA routes.
    # ------------------------------------------------------------------

    if FRONTEND_DIST.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str) -> Response:
            # An unrouted /api path is a MISSING ENDPOINT, never a client-side
            # route: answering it with index.html made every typo and every
            # renamed endpoint look like a 200 to the browser, so a fetch()
            # then died on "Unexpected token '<'" - or worse, a frontend
            # feature-probe ("use the real backend if it answers") concluded
            # the endpoint existed and fed HTML-shaped junk into its state.
            # Same for /ws: an HTTP GET there is a wrong protocol, not a page.
            first = path.split("/", 1)[0]
            if first in ("api", "ws"):
                return JSONResponse(
                    {"error": "not found", "detail": f"no endpoint at /{path}"},
                    status_code=404,
                )
            candidate = FRONTEND_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:

        @app.get("/")
        async def no_frontend() -> dict[str, str]:
            return {
                "message": "frontend not built - run: cd strands_robots/dashboard/frontend && npm install && npm run build",
                "api": "/api/health",
            }

    return app
