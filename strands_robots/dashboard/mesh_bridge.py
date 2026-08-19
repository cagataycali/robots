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
import io
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

PEER_STALE_S = 15.0  # presence heartbeat timeout before a card greys out

#: How long a peer may stay quiet before it is dropped from the fleet snapshot
#: entirely. A finished replay/collect peer stops heartbeating and never comes
#: back, so without this it lingers as a permanently stale ghost card.
PEER_TTL_S = float(os.getenv("STRANDS_DASHBOARD_PEER_TTL_S", "300"))


def prune_peers(
    peers: dict[str, dict[str, Any]],
    now: float,
    ttl: float = PEER_TTL_S,
    protected_ids: set[str] | frozenset[str] | None = None,
    stale_after: float = PEER_STALE_S,
) -> dict[str, dict[str, Any]]:
    """Flag quiet peers stale and drop the ones that aged past ``ttl``.

    Protected ids (peers backed by a LIVE managed local process) are always
    kept: their state stream may hiccup while the process is perfectly fine,
    and a card vanishing under a running robot is worse than a stale one.
    A non-positive ``ttl`` disables ageing out.
    """
    protected = protected_ids or frozenset()
    out: dict[str, dict[str, Any]] = {}
    for pid, entry in peers.items():
        age = now - entry.get("last_seen", 0)
        # Child sim peers are named "<parent>__<robot>"; they live and die with
        # the parent's process, so the parent's protection covers them.
        parent = pid.partition("__")[0]
        if ttl > 0 and age > ttl and pid not in protected and parent not in protected:
            continue
        out[pid] = {**entry, "stale": age > stale_after}
    return out


#: Transport low-pass filter on ``**/cmd`` (_zenoh_config.DEFAULT_MAX_CMD_BYTES).
#: Anything larger is dropped pre-deserialise and the sender only ever sees a
#: timeout, so we check before publishing and return a real error instead.
MAX_CMD_BYTES = int(os.getenv("STRANDS_MESH_MAX_CMD_BYTES", str(16 * 1024)))

#: How many fleet actions to keep for the activity panel.
ACTIVITY_CAP = 300


def route_task_target(target: str, cmd: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Route commands aimed at a child sim peer to its parent Simulation peer.

    Child sim peers ("<parent>__<robot>") stream state/cameras but cannot
    execute tasks themselves (upstream mesh._dispatch has no execute path on
    SimRobot children). The parent's _dispatch_sim_policy
    honors cmd["robot_name"], so we rewrite the target here. This is THE
    choke point: the REST API, the card ▶ button, and the
    fleet agent tool all go through it.
    """
    if "__" in target and not cmd.get("robot_name"):
        parent, _, robot_name = target.partition("__")
        if parent and robot_name:
            cmd = {**cmd, "robot_name": robot_name}
            target = parent
    return target, cmd


def command_succeeded(response: dict[str, Any] | None) -> bool:
    """Did a peer actually carry the command out?

    The wire is layered - ``{"type": "response", "result": {...}}`` for a
    dispatched command, ``{"type": "error", "error": ...}`` for a rejection -
    and a *successful* response can still carry ``result.ok == False`` (e.g.
    "peer exposes no stop_task; nothing was stopped"). Callers that only check
    for transport errors therefore report success while nothing stopped, which
    is how the ■ button came to lie.
    """
    if not isinstance(response, dict):
        return False
    if response.get("error") or response.get("type") == "error":
        return False
    if response.get("ok") is False:
        return False
    result = response.get("result")
    if isinstance(result, dict):
        if result.get("ok") is False:
            return False
        if str(result.get("status", "")).lower() in ("error", "failed"):
            return False
    return True


def _raw_to_jpeg(raw: bytes, shape: Any) -> tuple[bytes | None, str | None]:
    """Transcode raw pixel bytes to JPEG. Returns ``(jpeg, error)``.

    Only called for frames whose ``encoding`` is not JPEG. Needs the shape the
    publisher sent - without it the byte string is unreadable, and saying so is
    more useful than a black rectangle.
    """
    if not (isinstance(shape, (list, tuple)) and len(shape) in (2, 3)):
        return None, f"encoding is not jpeg and shape {shape!r} is unusable"
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None, "raw frame received but numpy/Pillow are not installed to transcode it"
    try:
        dims = tuple(int(d) for d in shape)
        expected = 1
        for d in dims:
            expected *= d
        if len(raw) != expected:
            return None, f"raw frame is {len(raw)} B but shape {dims} needs {expected} B"
        array = np.frombuffer(raw, dtype=np.uint8).reshape(dims)
        if array.ndim == 3 and array.shape[2] == 4:
            image = Image.fromarray(array, "RGBA").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array, "RGB")
        elif array.ndim == 2:
            image = Image.fromarray(array, "L")
        else:
            return None, f"unsupported raw frame shape {dims}"
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue(), None
    except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the sub
        return None, f"raw frame transcode failed: {type(exc).__name__}: {exc}"


def stop_outcome(response: dict[str, Any] | None) -> dict[str, Any]:
    """Classify one peer's answer to a stop into the three honest states.

    ``stopped`` / ``not_stopped`` (the peer answered but could not stop) /
    ``no_answer`` (timeout). ``emergency_stop()`` upstream makes exactly this
    distinction via ``peers_not_stopped``; the UI has to show it, because
    "unstoppable peer" and "peer offline" need different human reactions.
    """
    if not isinstance(response, dict):
        return {"state": "no_answer", "detail": "no response"}
    error = response.get("error")
    if error and "timeout" in str(error).lower():
        return {"state": "no_answer", "detail": str(error)}
    if command_succeeded(response):
        return {"state": "stopped", "detail": ""}
    result = response.get("result")
    detail = ""
    if isinstance(result, dict):
        detail = str(result.get("error") or result.get("status") or "")
    return {"state": "not_stopped", "detail": detail or str(error or "refused")}


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

        # Signed safety rail (lazy - see _safety_mesh)
        self._safety: Any | None = None
        self._safety_lock = threading.Lock()

        # RPC correlation (mirrors Mesh.send)
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._rpc_lock = threading.Lock()

        # Fleet activity: every command this dashboard issued + safety
        # envelopes seen on the wire. Cheap forensics for "who moved that arm?"
        self.activity: deque[dict[str, Any]] = deque(maxlen=ACTIVITY_CAP)
        self._activity_lock = threading.Lock()

        # Resolved endpoints of the live session, for /api/mesh/config.
        self._endpoints: dict[str, Any] = {}

        # Set by the server to a callable returning peer ids that must never be
        # aged out (LIVE managed local processes). Optional by design: the
        # bridge stays usable standalone.
        self.protected_peer_ids: Any | None = None

    def _protected_peer_ids(self) -> frozenset[str]:
        if self.protected_peer_ids is None:
            return frozenset()
        try:
            return frozenset(self.protected_peer_ids())
        except Exception as exc:  # never let a bad hook break the snapshot
            logger.warning("[mesh] protected peer lookup failed (%r)", exc)
            return frozenset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Join the mesh. Returns False when zenoh is unavailable."""
        from strands_robots.dashboard import settings
        from strands_robots.mesh.session import get_session

        # ZENOH_CONNECT / ZENOH_LISTEN / STRANDS_MESH_PORT are read *inside*
        # get_session(), so remote-endpoint settings have to be in the
        # environment before this call - not after.
        settings.apply_mesh_env()

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
        self._endpoints = self._read_endpoints()
        logger.info("MeshBridge online as %s", self.peer_id)
        return True

    def _read_endpoints(self) -> dict[str, Any]:
        """What the live session is actually talking to."""
        from strands_robots.dashboard import settings

        info: dict[str, Any] = {
            "connect": settings.as_list(os.getenv("ZENOH_CONNECT")),
            "listen": settings.as_list(os.getenv("ZENOH_LISTEN")),
            "port": os.getenv("STRANDS_MESH_PORT", "7447"),
            "backend": os.getenv("STRANDS_MESH_BACKEND", "zenoh"),
        }
        try:
            from strands_robots.mesh._zenoh_config import resolve_auth_mode

            info["auth_mode"] = resolve_auth_mode()
        except Exception:  # noqa: BLE001 - introspection only
            info["auth_mode"] = "unknown"
        return info

    def mesh_info(self) -> dict[str, Any]:
        """Mesh posture for /api/mesh/config and the settings panel."""
        from strands_robots.dashboard import settings

        local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")
        camera_hz = os.getenv("STRANDS_MESH_CAMERA_HZ")
        info = {
            **self._endpoints,
            "online": self._running,
            "peer_id": self.peer_id,
            "peers": len(self.peers),
            "live_peers": len(self.live_peers()),
            "local_dev": local_dev,
            # STRANDS_MESH_LOCAL_DEV=1 runs the entire mesh with auth "none";
            # _build_config logs "WIRE SECURITY DISABLED". Surface it so a lab
            # posture can't be mistaken for a secured one.
            "wire_security": "DISABLED (local dev)" if local_dev else self._endpoints.get("auth_mode"),
            "camera_hz": float(camera_hz) if camera_hz else 0.0,
            "settings": settings.load()["mesh"],
            "multicast": os.getenv("STRANDS_MESH_MULTICAST", ""),
            "max_cmd_bytes": MAX_CMD_BYTES,
        }
        try:
            from strands_robots.mesh.security import _policy_type_allowlist

            info["policy_allow"] = sorted(_policy_type_allowlist())
        except Exception:  # noqa: BLE001
            info["policy_allow"] = []
        return info

    def restart(self) -> bool:
        """Re-open the mesh session against the current settings.

        The upstream session is a ref-counted module singleton, so a reopen
        only picks up new endpoints once every consumer has released it -
        stopping the bridge is what drops our reference.
        """
        loop = self._loop
        if loop is None:
            return False
        self.stop()
        with self._peers_lock:
            self.peers.clear()
        with self._frames_lock:
            self.frames.clear()
        ok = self.start(loop)
        self.record_activity("mesh", "restart", detail=self._endpoints, ok=ok)
        self._emit({"type": "mesh_reconfigured", "ok": ok, "mesh": self.mesh_info()})
        return ok

    def stop(self) -> None:
        self._running = False
        if self._safety is not None:
            try:
                self._safety.stop()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self._safety = None
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
        meta: dict[str, Any] = {
            "t": data.get("t"),
            "shape": data.get("shape"),
            "encoding": data.get("encoding"),
        }
        # A peer may publish raw pixel bytes instead of JPEG. Handing those to
        # an <img> (or serving them as image/jpeg) produces a black tile that
        # looks exactly like a dead camera, so transcode here and say so when we
        # cannot.
        if str(meta["encoding"] or "jpeg").lower() not in ("jpeg", "jpg"):
            raw, error = _raw_to_jpeg(raw, meta.get("shape"))
            meta["converted"] = error is None
            if error:
                meta["error"] = error
        meta["displayable"] = raw is not None
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
        self.record_activity("safety", kind, detail=data, ok=True)
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

    # ------------------------------------------------------------------
    # Signed safety rail (A6). The dashboard's raw zenoh session can only
    # broadcast per-peer stop commands; the SIGNED strands/safety/estop
    # envelope (SourceInfo zid binding + HMAC resume proof + fleet-wide
    # lockout) needs a Mesh instance. Mesh accepts robot=None - a robot-less
    # gateway signs and fans out exactly like a robot peer, it just has
    # nothing of its own to stop.
    # ------------------------------------------------------------------

    def _safety_mesh(self) -> Any | None:
        """Lazily start the robot-less Mesh used for signed safety envelopes."""
        with self._safety_lock:
            if self._safety is not None and getattr(self._safety, "alive", False):
                return self._safety
            try:
                from strands_robots.mesh.core import Mesh

                m = Mesh(None, peer_id=f"{self.peer_id}-safety", peer_type="gateway")
                m.start()
                if not m.alive:
                    return None
                self._safety = m
                logger.info("signed safety rail online as %s", m.peer_id)
                return m
            except Exception as exc:  # noqa: BLE001 - rail is enrichment over broadcast-stop
                logger.warning("signed safety rail unavailable: %s", exc)
                return None

    def signed_estop(self) -> dict[str, Any]:
        """Fleet e-stop over the SIGNED rail.

        Publishes the strands/safety/estop envelope (SourceInfo zid binding)
        which engages the LOCKOUT on every listening peer - they refuse all
        non-status commands until a proofed resume - and aggregates the
        per-peer stop responses the mesh's own broadcast collected.
        """
        m = self._safety_mesh()
        if m is None:
            return {"signed": False, "error": "safety mesh unavailable"}
        responses = m.emergency_stop()
        return {
            "signed": True,
            "issuer": m.peer_id,
            "responses": responses,
            "lockout_engaged": True,
        }

    def signed_resume(self, override_code: str) -> dict[str, Any]:
        """Clear the fleet lockout with the operator override code.

        The local safety mesh verifies the code (brute-force throttled),
        clears its own lockout, and publishes the HMAC-proofed resume
        envelope every peer independently re-verifies.
        """
        m = self._safety_mesh()
        if m is None:
            return {"signed": False, "error": "safety mesh unavailable"}
        res = m._resume_lockout(override_code)
        return {"signed": True, "issuer": m.peer_id, **(res or {})}

    def send_cmd(
        self,
        target: str,
        cmd: dict[str, Any],
        timeout: float = 30.0,
        *,
        source: str = "api",
    ) -> dict[str, Any]:
        """Send a command to a peer and wait for its response (blocking)."""
        from strands_robots.mesh.session import put

        if not self._running:
            return {"error": "mesh offline", "ok": False}
        turn = uuid.uuid4().hex
        envelope = {
            "sender_id": self.peer_id,
            "turn_id": turn,
            "command": cmd,
            "timestamp": time.time(),
        }
        # The transport silently drops cmd messages over the low-pass cap, so
        # the caller would otherwise see a bare timeout with nothing to act on.
        size = len(json.dumps(envelope, default=str).encode())
        if size > MAX_CMD_BYTES:
            result = {
                "ok": False,
                "error": f"command too large: {size} B > transport cap {MAX_CMD_BYTES} B "
                         "(raise STRANDS_MESH_MAX_CMD_BYTES on every peer, or shrink the payload)",
            }
            self.record_activity(source, cmd.get("action", "?"), target=target, detail=result, ok=False)
            return result

        evt = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = evt
        started = time.time()
        try:
            put(f"strands/{target}/cmd", envelope)
            if not evt.wait(timeout):
                result = {"error": f"timeout after {timeout:g}s", "turn_id": turn, "ok": False}
            else:
                with self._rpc_lock:
                    result = self._responses.pop(turn, {"error": "response lost", "ok": False})
            self.record_activity(
                source,
                cmd.get("action", "?"),
                target=target,
                detail={"instruction": cmd.get("instruction"), "provider": cmd.get("policy_provider")},
                ok=command_succeeded(result),
                result=result,
                elapsed=time.time() - started,
            )
            return result
        finally:
            with self._rpc_lock:
                self._pending.pop(turn, None)
                self._responses.pop(turn, None)

    async def send_cmd_async(
        self,
        target: str,
        cmd: dict[str, Any],
        timeout: float = 30.0,
        *,
        source: str = "api",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.send_cmd, target, cmd, timeout, source=source)

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    def record_activity(
        self,
        source: str,
        action: str,
        *,
        target: str = "",
        detail: Any = None,
        ok: bool | None = None,
        result: Any = None,
        elapsed: float | None = None,
    ) -> None:
        entry = {
            "t": time.time(),
            "source": source,
            "action": action,
            "target": target,
            "ok": ok,
            "detail": detail,
            "elapsed": round(elapsed, 3) if elapsed is not None else None,
        }
        if result is not None:
            entry["result"] = json.dumps(result, default=str)[:400]
        with self._activity_lock:
            self.activity.append(entry)
        self._emit({"type": "activity", "data": entry})

    def activity_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._activity_lock:
            items = list(self.activity)
        return items[-limit:][::-1]

    # ------------------------------------------------------------------
    # Snapshot for initial page load
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        protected = self._protected_peer_ids()
        with self._peers_lock:
            peers = prune_peers(self.peers, now, PEER_TTL_S, protected)
            # Forget the aged-out peers for good: keeping them in self.peers
            # only feeds the same ghosts back on every later snapshot.
            for pid in set(self.peers) - set(peers):
                self.peers.pop(pid, None)
        return {
            "type": "snapshot",
            "dashboard_peer_id": self.peer_id,
            "peers": peers,
            "mesh": self.mesh_info(),
            "t": now,
        }

    def live_peers(self) -> list[str]:
        """Peer ids with a fresh presence heartbeat.

        Fleet-wide operations use this instead of every id we have ever seen:
        gathering stops from dead peers just blocks the response for the full
        RPC timeout and reports nothing useful.
        """
        now = time.time()
        with self._peers_lock:
            return [
                pid for pid, entry in self.peers.items()
                if (now - entry.get("last_seen", 0)) <= PEER_STALE_S
            ]

    def latest_frame(self, peer_id: str, cam: str) -> dict[str, Any] | None:
        with self._frames_lock:
            return self.frames.get((peer_id, cam))
