"""FastAPI app for the Strands Robots Dashboard.

Endpoints:
    GET  /api/health                     liveness + mesh status (always public)
    GET  /api/fleet                      current fleet snapshot
    GET  /api/robots/registry            registered robot names (spawnable)
    GET  /api/policies                   full provider catalog (run-form schema)
    POST /api/policies/validate          pre-flight a provider config
    POST /api/robots/{peer}/task         start a task on a peer
    POST /api/robots/{peer}/stop         stop the running task on a peer
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
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from strands_robots.dashboard import config_api, settings
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
        allowed = settings.get("security", "cors_origins", []) or []
        if "*" in allowed or origin.rstrip("/") in {str(a).rstrip("/") for a in allowed}:
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


def create_app(bridge: MeshBridge | None = None) -> FastAPI:
    app = FastAPI(title="strands-robots dashboard")
    origins = settings.get("security", "cors_origins", []) or []
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
    # A peer with a LIVE managed local process is never aged out of the fleet
    # snapshot, even if its state stream goes quiet.
    app.state.bridge.protected_peer_ids = lambda: {
        pid for pid, m in list(app.state.devices.robots.items()) if m.alive()
    }

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
                await asyncio.to_thread(watcher.poll)
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
            "t": time.time(),
        }

    @app.get("/api/fleet")
    async def fleet() -> dict[str, Any]:
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

    @app.get("/api/robots/{peer_id}/teleop")
    async def teleop_status(peer_id: str) -> dict[str, Any]:
        """Live teleop health for one peer: publisher/receiver rates, drops,
        slew rejections - the counters InputPublisher/InputReceiver already
        keep. Works on hardware AND sim peers (TeleopMixin lift)."""
        result = await app.state.bridge.send_cmd_async(peer_id, {"action": "teleop_status"}, timeout=10.0)
        return {"peer_id": peer_id, "result": result}

    @app.post("/api/robots/{peer_id}/teleop/receive")
    async def teleop_receive(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Point a follower (real or sim twin) at a leader's input stream.

        With the TeleopMixin lift a SIM twin can follow a REAL leader arm -
        practice-on-the-twin before metal.
        """
        source = (body.get("source_peer_id") or "").strip()
        if not source:
            raise HTTPException(422, "source_peer_id required")
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
        return await asyncio.to_thread(
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
        return await asyncio.to_thread(
            app.state.devices.replay,
            repo_id,
            int(body.get("episode", 0)),
            body.get("root"),
            float(body.get("speed", 1.0)),
            body.get("robot_name") or "so101",
        )

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

        return await asyncio.to_thread(training.submit, body)

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

        return await asyncio.to_thread(checkpoints.search, q, min(int(limit), 40))

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

        return {
            "peer_id": peer_id,
            "routed_to": target if target != peer_id else None,
            "mirrored_to_twin": mirrored,
            # A response can arrive and still say ok=False. The UI needs one
            # honest boolean, not a nested shape it has to re-guess.
            "ok": command_succeeded(result),
            "timeout_s": timeout_s,
            "result": result,
        }

    @app.post("/api/robots/{peer_id}/stop")
    async def stop_task(peer_id: str) -> dict[str, Any]:
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

    @app.get("/api/devices")
    async def devices(refresh: bool = False) -> dict[str, Any]:
        """Local USB serial ports (servo buses) + cameras + managed robots.

        Camera probe results are cached ~30s and indices owned by running
        robots are never re-opened; ``?refresh=1`` forces a
        fresh probe of the unclaimed indices.
        """
        return await asyncio.to_thread(app.state.devices.devices, refresh)

    @app.post("/api/devices/spawn")
    async def spawn(body: dict[str, Any]) -> dict[str, Any]:
        robot_name = body.get("robot_name")
        if not robot_name:
            raise HTTPException(422, "robot_name required")
        return await asyncio.to_thread(
            app.state.devices.spawn,
            robot_name,
            body.get("mode", "sim"),
            body.get("peer_id"),
            body.get("port"),
            body.get("cameras"),
            body.get("robot_id"),
        )

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
        return await asyncio.to_thread(app.state.devices.despawn, peer_id)

    @app.get("/api/devices/logs/{peer_id}")
    async def device_logs(peer_id: str) -> dict[str, Any]:
        """Child-process output for one managed robot (ring buffer)."""
        return app.state.devices.logs(peer_id)

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
    async def calibration_view(name: str, device_type: str = "robots") -> dict[str, Any]:
        from strands_robots.tools.lerobot_calibrate import lerobot_calibrate

        res = await asyncio.to_thread(lerobot_calibrate, "view", name, device_type)
        return {"status": res.get("status"), "text": (res.get("content") or [{}])[0].get("text", "")}

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

        from strands_robots.dashboard.agent_bridge import run_turn_blocking

        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = {"type": "chat", "text": raw}
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                    continue
                prompt = (msg.get("text") or "").strip()
                if not prompt:
                    continue
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
        async def spa(path: str) -> FileResponse:
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
