"""FastAPI app for the Strands Robots Dashboard.

Endpoints (Phase 0/1):
    GET  /api/health                     liveness + mesh status
    GET  /api/fleet                      current fleet snapshot
    GET  /api/robots/registry            registered robot names (spawnable)
    POST /api/robots/{peer}/task         start a task (execute) on a peer
    POST /api/robots/{peer}/stop         stop the running task on a peer
    POST /api/safety/estop               fleet-wide emergency stop
    GET  /api/frame/{peer}/{cam}         latest JPEG frame (poll)
    WS   /ws/mesh                        live event stream (presence/state/stream/camera_meta/safety)
    WS   /ws/camera/{peer}/{cam}         binary JPEG stream for one camera tile
    /                                    static PWA (frontend/dist)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from strands_robots.dashboard.device_manager import DeviceManager
from strands_robots.dashboard.mesh_bridge import MeshBridge

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def create_app(bridge: MeshBridge | None = None) -> FastAPI:
    app = FastAPI(title="strands-robots dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        from strands_robots.policies import list_providers

        try:
            return {"providers": list_providers()}
        except Exception as exc:
            raise HTTPException(500, f"policy registry unavailable: {exc}") from exc

    @app.post("/api/robots/{peer_id}/task")
    async def start_task(peer_id: str, body: dict[str, Any]) -> dict[str, Any]:
        instruction = (body.get("instruction") or "").strip()
        if not instruction:
            raise HTTPException(422, "instruction required")
        cmd = {
            "action": "execute",
            "instruction": instruction,
            "policy_provider": body.get("policy_provider", "mock"),
            "duration": float(body.get("duration", 30.0)),
        }
        for opt in ("policy_port", "policy_host", "policy_config", "robot_name"):
            if body.get(opt) is not None:
                cmd[opt] = body[opt]
        # Child sim peers can't execute themselves:
        # route "<parent>__<robot>" to the parent with robot_name here, so
        # the card ▶ button and every API caller get the fix - not just the
        # agent's fleet tool.
        from strands_robots.dashboard.mesh_bridge import route_task_target

        target, cmd = route_task_target(peer_id, cmd)
        timeout_s = float(body.get("timeout", 60.0))
        # Twin mirroring: fire the same instruction at '<peer>-twin' when one
        # is live (fire-and-forget; twin progress streams on the mesh).
        twin_id = f"{peer_id}-twin"
        twin = app.state.bridge.peers.get(twin_id)
        mirrored = bool(twin and not twin.get("stale"))
        if mirrored:
            t_target, t_cmd = route_task_target(twin_id, dict(cmd))
            asyncio.create_task(app.state.bridge.send_cmd_async(t_target, t_cmd, timeout=timeout_s))
        result = await app.state.bridge.send_cmd_async(target, cmd, timeout=timeout_s)
        return {
            "peer_id": peer_id,
            "routed_to": target if target != peer_id else None,
            "mirrored_to_twin": mirrored,
            "result": result,
        }

    @app.post("/api/robots/{peer_id}/stop")
    async def stop_task(peer_id: str) -> dict[str, Any]:
        result = await app.state.bridge.send_cmd_async(peer_id, {"action": "stop"}, timeout=10.0)
        return {"peer_id": peer_id, "result": result}

    @app.post("/api/safety/estop")
    async def estop() -> dict[str, Any]:
        """Fleet-wide stop: broadcast {action: stop} to every known peer.

        Note: the signed strands/safety/estop envelope requires a Mesh
        instance with a robot; Phase 0 sends per-peer stop commands. The
        full lockout-engaging e-stop lands when the dashboard embeds a
        local sim peer (Phase 3) or we lift envelope signing into the bridge.
        """
        bridge: MeshBridge = app.state.bridge
        peers = list(bridge.peers.keys())
        results = await asyncio.gather(
            *(bridge.send_cmd_async(p, {"action": "stop"}, timeout=5.0) for p in peers),
            return_exceptions=True,
        )
        return {
            "stopped": {
                p: (r if isinstance(r, dict) else {"error": str(r)})
                for p, r in zip(peers, results)
            }
        }

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
        try:
            while True:
                f = bridge.latest_frame(peer_id, cam)
                if f is not None and f.get("t") != last_t:
                    last_t = f.get("t")
                    await ws.send_bytes(f["jpeg"])
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
