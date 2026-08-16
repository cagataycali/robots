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
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from strands_robots.dashboard import config_api, settings
from strands_robots.dashboard.device_manager import DeviceManager
from strands_robots.dashboard.mesh_bridge import MeshBridge, stop_outcome

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

#: Reachable without a token: liveness (so a client can discover that auth is
#: required at all) and the static shell (which renders the token prompt).
PUBLIC_PATHS = {"/api/health"}


class TokenAuthMiddleware:
    """Bearer-token gate for /api and /ws, as raw ASGI.

    Raw ASGI rather than ``BaseHTTPMiddleware`` because WebSocket scopes never
    reach an HTTP middleware - and /ws/mesh, /ws/chat and /ws/voice are exactly
    the endpoints that drive motors and spend money.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

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

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        guarded = path.startswith("/api") or path.startswith("/ws")
        token = settings.get("security", "auth_token")
        if not guarded or not token or path in PUBLIC_PATHS or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if hmac.compare_digest(self._presented(scope), str(token)):
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
    origins = settings.get("security", "cors_origins", ["*"]) or ["*"]
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

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        app.state.mesh_online = await asyncio.to_thread(app.state.bridge.start, loop)
        # Hand the mesh gateway to the fleet agent (chat + voice).
        from strands_robots.dashboard import agent_bridge as _ab

        _ab.set_bridge(app.state.bridge)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
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
        return {
            "targeted": peers,
            "stale_skipped": stale,
            "counts": counts,
            # True only when every live peer confirmed. Anything else and the
            # UI must keep shouting.
            "all_stopped": bool(peers) and counts["stopped"] == len(peers),
            "stopped": per_peer,
        }

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
                _threading.Thread(target=run_turn_blocking, args=(prompt, q), daemon=True).start()
                while True:
                    ev = await asyncio.to_thread(q.get)
                    if ev.get("type") == "__END__":
                        break
                    await ws.send_text(json.dumps(ev))
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
