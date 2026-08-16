"""Zenoh mesh <-> asyncio bridge for the dashboard.

Joins the shared mesh session (``strands_robots.mesh.session.get_session`` -
inherits namespace / TLS / ACL config) and:

* subscribes ``strands/*/presence``, ``strands/*/state``, ``strands/*/stream``,
  ``strands/*/camera/**``, ``strands/safety/**``
* maintains an in-memory fleet snapshot (peers keyed by peer_id)
* fans events out to any number of async consumers (WebSocket handlers)
* publishes commands to peers (``strands/<peer>/cmd``) with RPC correlation
  on ``strands/<dash_id>/response/**`` - mirroring ``Mesh.send()``.

Camera frames are kept OUT of the JSON event stream: they are stored in a
latest-frame cache and served/fanned-out as binary (see server.py) so a
phone on wifi is not decoding base64 JSON at 30 Hz.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

PEER_STALE_S = 15.0  # presence heartbeat timeout before a card greys out


def route_task_target(target: str, cmd: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Route commands aimed at a child sim peer to its parent Simulation peer.

    Child sim peers ("<parent>__<robot>") stream state/cameras but cannot
    execute tasks themselves (upstream mesh._dispatch has no execute path on
    SimRobot children - BUGS.md #11). The parent's _dispatch_sim_policy
    honors cmd["robot_name"], so we rewrite the target here. This is THE
    choke point (BUGS.md #13): the REST API, the card ▶ button, and the
    fleet agent tool all go through it.
    """
    if "__" in target and not cmd.get("robot_name"):
        parent, _, robot_name = target.partition("__")
        if parent and robot_name:
            cmd = {**cmd, "robot_name": robot_name}
            target = parent
    return target, cmd


class MeshBridge:
    """Dashboard-side mesh peer. One instance per server process."""

    def __init__(self, peer_id: str | None = None) -> None:
        self.peer_id = peer_id or f"dashboard-{socket.gethostname().split('.')[0]}-{uuid.uuid4().hex[:4]}"
        self._session: Any | None = None
        self._subs: list[Any] = []
        self._running = False

        # Fleet snapshot: peer_id -> {presence, state, stream, last_seen, cameras:{name:{t,shape}}}
        self.peers: dict[str, dict[str, Any]] = {}
        self._peers_lock = threading.Lock()

        # Latest camera frames: (peer_id, cam) -> {"t": float, "jpeg": bytes, "shape": [...]}
        self.frames: dict[tuple[str, str], dict[str, Any]] = {}
        self._frames_lock = threading.Lock()

        # Async fan-out. Subscribers get JSON-able event dicts.
        self._queues: set[asyncio.Queue] = set()
        self._queues_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

        # RPC correlation (mirrors Mesh.send)
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._rpc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Join the mesh. Returns False when zenoh is unavailable."""
        from strands_robots.mesh.session import get_session

        self._loop = loop
        session = get_session()
        if session is None:
            logger.warning("Mesh session unavailable (is eclipse-zenoh installed?) - dashboard runs offline")
            return False
        self._session = session
        self._running = True

        sub = session.declare_subscriber
        self._subs = [
            sub("strands/*/presence", self._on_presence),
            sub("strands/*/state", self._on_state),
            sub("strands/*/stream", self._on_stream),
            sub("strands/*/camera/**", self._on_camera),
            sub("strands/safety/estop", self._on_safety),
            sub("strands/safety/resume", self._on_safety),
            sub(f"strands/{self.peer_id}/response/**", self._on_response),
        ]
        logger.info("MeshBridge online as %s", self.peer_id)
        return True

    def stop(self) -> None:
        self._running = False
        for s in self._subs:
            with contextlib.suppress(Exception):
                s.undeclare()
        self._subs.clear()
        if self._session is not None:
            from strands_robots.mesh.session import release_session

            with contextlib.suppress(Exception):
                release_session()
            self._session = None

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def attach_queue(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._queues_lock:
            self._queues.add(q)
        return q

    def detach_queue(self, q: asyncio.Queue) -> None:
        with self._queues_lock:
            self._queues.discard(q)

    def _emit(self, event: dict[str, Any]) -> None:
        """Push an event to all consumer queues (thread -> loop safe)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._queues_lock:
            queues = list(self._queues)
        for q in queues:
            def _put(q=q):
                if q.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        q.get_nowait()  # drop oldest - dashboards want latest
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)
            loop.call_soon_threadsafe(_put)

    # ------------------------------------------------------------------
    # Zenoh callbacks (zenoh worker threads)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(sample: Any) -> dict[str, Any] | None:
        try:
            data = json.loads(sample.payload.to_bytes().decode())
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _touch_peer(self, peer_id: str) -> dict[str, Any]:
        with self._peers_lock:
            entry = self.peers.setdefault(peer_id, {"peer_id": peer_id})
            entry["last_seen"] = time.time()
            return entry

    def _on_presence(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("robot_id")
        if not isinstance(peer_id, str) or peer_id == self.peer_id:
            return
        entry = self._touch_peer(peer_id)
        entry["presence"] = data
        self._emit({"type": "presence", "peer_id": peer_id, "data": data})

    def _on_state(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return
        entry = self._touch_peer(peer_id)
        entry["state"] = data
        self._emit({"type": "state", "peer_id": peer_id, "data": data})

    def _on_stream(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return
        entry = self._touch_peer(peer_id)
        entry["stream"] = data
        self._emit({"type": "stream", "peer_id": peer_id, "data": data})

    def _on_camera(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        cam = data.get("cam")
        encoded = data.get("data")
        if not (isinstance(peer_id, str) and isinstance(cam, str) and isinstance(encoded, str)):
            return
        import base64

        try:
            raw = base64.b64decode(encoded)
        except Exception:
            return
        meta = {"t": data.get("t"), "shape": data.get("shape"), "encoding": data.get("encoding")}
        with self._frames_lock:
            self.frames[(peer_id, cam)] = {"jpeg": raw, **meta}
        entry = self._touch_peer(peer_id)
        cams = entry.setdefault("cameras", {})
        cams[cam] = meta
        # Lightweight notification (no pixels) so the UI knows a frame arrived.
        self._emit({"type": "camera_meta", "peer_id": peer_id, "cam": cam, "data": meta})

    def _on_safety(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        key = str(getattr(sample, "key_expr", ""))
        kind = "estop" if key.endswith("estop") else "resume"
        self._emit({"type": "safety", "kind": kind, "data": data})

    def _on_response(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return
        with self._rpc_lock:
            evt = self._pending.get(turn)
            if evt is not None:
                self._responses[turn] = data
                evt.set()

    # ------------------------------------------------------------------
    # Commands (dashboard -> robot). Human-initiated; bypasses LLM HITL
    # gates by design (the human IS the loop - they clicked the button).
    # ------------------------------------------------------------------

    def send_cmd(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send a command to a peer and wait for its response (blocking)."""
        from strands_robots.mesh.session import put

        if not self._running:
            return {"error": "mesh offline"}
        turn = uuid.uuid4().hex
        evt = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = evt
        try:
            put(
                f"strands/{target}/cmd",
                {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()},
            )
            if not evt.wait(timeout):
                return {"error": f"timeout after {timeout:g}s", "turn_id": turn}
            with self._rpc_lock:
                return self._responses.pop(turn, {"error": "response lost"})
        finally:
            with self._rpc_lock:
                self._pending.pop(turn, None)
                self._responses.pop(turn, None)

    async def send_cmd_async(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        return await asyncio.to_thread(self.send_cmd, target, cmd, timeout)

    # ------------------------------------------------------------------
    # Snapshot for initial page load
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._peers_lock:
            peers = {
                pid: {**entry, "stale": (now - entry.get("last_seen", 0)) > PEER_STALE_S}
                for pid, entry in self.peers.items()
            }
        return {"type": "snapshot", "dashboard_peer_id": self.peer_id, "peers": peers, "t": now}

    def latest_frame(self, peer_id: str, cam: str) -> dict[str, Any] | None:
        with self._frames_lock:
            return self.frames.get((peer_id, cam))
