"""Dashboard HTTP + WebSocket server (aiohttp, single port).

Architecture::

    Robot()/Sim() --> Zenoh mesh <-- MeshObserver --(thread cb)--> asyncio queue
                                                                        |
                                          browser <== WebSocket ==  fan-out task
                                          browser  ==> WS cmd     ==> observer.send

One aiohttp app serves:
    GET  /            -> static index.html
    GET  /static/*    -> static assets
    GET  /ws          -> WebSocket (state push + command intake)
    GET  /healthz     -> liveness probe

The mesh observer runs on Zenoh's own callback threads. We bridge those into
the asyncio loop via ``loop.call_soon_threadsafe`` onto an ``asyncio.Queue``,
then a single fan-out task pushes each event to every connected WS client.
Browser-issued commands flow back out through the observer's mesh RPC path,
which is validated and audited by the mesh security layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from strands_robots.dashboard.observer import MeshObserver

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Bounded so a slow/dead browser can't make us buffer the whole mesh history.
_EVENT_QUEUE_MAX = 2000


class DashboardServer:
    """Owns the observer, the WS client set, and the asyncio bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7860) -> None:
        self.host = host
        self.port = port
        self._clients: weakref.WeakSet[web.WebSocketResponse] = weakref.WeakSet()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._observer = MeshObserver(on_event=self._on_mesh_event)
        self._fanout_task: asyncio.Task[None] | None = None
        self._peer_poll_task: asyncio.Task[None] | None = None

    # -- mesh callback (runs on a Zenoh thread) -------------------------

    def _on_mesh_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Thread-safe hand-off from a Zenoh callback into the asyncio loop."""
        if self._loop is None or self._queue is None:
            return
        msg = {"type": kind, "data": payload, "ts": time.time()}

        def _enqueue() -> None:
            assert self._queue is not None
            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                # Drop oldest to stay live rather than blocking the mesh.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(msg)
                except Exception:  # noqa: BLE001
                    pass

        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            # Loop is shutting down.
            pass

    # -- asyncio tasks --------------------------------------------------

    async def _fanout(self) -> None:
        """Drain the event queue and broadcast to all WS clients."""
        assert self._queue is not None
        while True:
            msg = await self._queue.get()
            await self._broadcast(msg)

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(msg, default=str)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _peer_poll(self) -> None:
        """Periodically push the full peer snapshot (covers prune/leave)."""
        while True:
            await asyncio.sleep(2.0)
            peers = self._observer.peers()
            await self._broadcast(
                {"type": "peers", "data": {"peers": peers}, "ts": time.time()}
            )

    # -- HTTP handlers --------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.StreamResponse:
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            return web.Response(text="dashboard index.html missing", status=500)
        return web.FileResponse(index)

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "mesh_alive": self._observer.alive,
                "peer_id": self._observer.peer_id,
                "clients": len(self._clients),
            }
        )

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info("ws client connected (%d total)", len(self._clients))

        # Send a hello + immediate peer snapshot so the UI paints instantly.
        await ws.send_str(
            json.dumps(
                {
                    "type": "hello",
                    "data": {
                        "peer_id": self._observer.peer_id,
                        "mesh_alive": self._observer.alive,
                        "peers": self._observer.peers(),
                    },
                    "ts": time.time(),
                }
            )
        )

        try:
            async for raw in ws:
                if raw.type == WSMsgType.TEXT:
                    await self._on_ws_message(ws, raw.data)
                elif raw.type == WSMsgType.ERROR:
                    logger.debug("ws error: %s", ws.exception())
        finally:
            self._clients.discard(ws)
            logger.info("ws client disconnected (%d left)", len(self._clients))
        return ws

    async def _on_ws_message(self, ws: web.WebSocketResponse, raw: str) -> None:
        """Handle a command from the browser. Runs mesh RPC off-thread."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_str(json.dumps({"type": "error", "data": {"error": "bad json"}}))
            return

        action = msg.get("action")
        req_id = msg.get("req_id")

        # Mesh RPC calls block (event.wait); run them in a thread so we don't
        # stall the event loop / other clients.
        def _do() -> Any:
            if action == "peers":
                return {"peers": self._observer.peers()}
            if action == "emergency_stop":
                return {"responses": self._observer.emergency_stop()}
            if action == "send":
                target = msg.get("target", "")
                cmd = msg.get("cmd", {})
                if not target or not isinstance(cmd, dict):
                    return {"error": "send requires target + cmd"}
                return {"response": self._observer.send(target, cmd, timeout=msg.get("timeout", 10.0))}
            if action == "broadcast":
                cmd = msg.get("cmd", {})
                if not isinstance(cmd, dict):
                    return {"error": "broadcast requires cmd"}
                return {"responses": self._observer.broadcast(cmd, timeout=msg.get("timeout", 5.0))}
            if action == "teleop_start":
                target = msg.get("target", "")
                if not target:
                    return {"error": "teleop_start requires target"}
                return {"response": self._observer.start_teleop(target, msg.get("device"))}
            if action == "teleop_frame":
                target = msg.get("target", "")  # informational; frame goes on dashboard topic
                frame = msg.get("frame", {})
                if not isinstance(frame, dict):
                    return {"error": "teleop_frame requires frame dict"}
                self._observer.teleop_frame(frame, msg.get("device"), msg.get("events"))
                return {"ok": True}
            if action == "teleop_stop":
                target = msg.get("target", "")
                if not target:
                    return {"error": "teleop_stop requires target"}
                return {"response": self._observer.stop_teleop(target, msg.get("device"))}
            return {"error": f"unknown action: {action}"}

        try:
            result = await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the socket
            result = {"error": str(exc)}

        await ws.send_str(
            json.dumps({"type": "ack", "req_id": req_id, "action": action, "data": result}, default=str)
        )

    # -- lifecycle ------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/ws", self._handle_ws)
        if _STATIC_DIR.exists():
            app.router.add_static("/static/", _STATIC_DIR, show_index=False)
        return app

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)

        alive = self._observer.start()
        if not alive:
            logger.warning(
                "Dashboard mesh observer is not alive. The UI will still load, "
                "but no live mesh data will arrive until a mesh transport is up."
            )

        self._fanout_task = asyncio.create_task(self._fanout())
        self._peer_poll_task = asyncio.create_task(self._peer_poll())

        app = self._build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        url = f"http://{self.host}:{self.port}"
        logger.info("Strands Robots dashboard listening on %s", url)
        print(f"\n  Strands Robots dashboard:  {url}")
        print(f"  Mesh peer id:              {self._observer.peer_id}")
        print(f"  Mesh alive:                {self._observer.alive}\n")

        try:
            await asyncio.Event().wait()  # run forever
        finally:
            for task in (self._fanout_task, self._peer_poll_task):
                if task is not None:
                    task.cancel()
            self._observer.stop()
            await runner.cleanup()

    def run(self) -> None:
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            print("\n  dashboard stopped.")


def start_dashboard(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Start the dashboard server (blocking).

    Parameters
    ----------
    host:
        Bind address. Defaults to ``127.0.0.1`` — the dashboard exposes robot
        teleop and e-stop, so it binds loopback-only unless an operator opts
        into network exposure with ``host="0.0.0.0"``.
    port:
        TCP port. Defaults to ``7860``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    DashboardServer(host=host, port=port).run()
