"""Per-robot mesh component — presence broadcast and peer discovery.

A :class:`Mesh` is a small, embeddable component owned by a single
:class:`~strands_robots.simulation.Simulation` or
:class:`~strands_robots.hardware_robot.Robot` instance.  It puts the robot on
the Zenoh mesh, broadcasts a 2 Hz presence heartbeat so other peers can
discover it, and publishes a 10 Hz numeric state stream that other peers can
subscribe to.

This module is intentionally minimal:

* It owns *no* RPC machinery (``send`` / ``broadcast`` / ``tell`` land in PR3).
* It owns *no* user subscription API or VLA stream publishing (PR4).
* It owns *no* emergency-stop / audit logging (PR5).

What lives here are exactly the two things every mesh peer needs to be
*discoverable*: a presence loop and a state loop, plus the lifecycle plumbing
that wires them to the shared :mod:`strands_robots.mesh_session` singleton.

Topic schema
------------
``strands/{peer_id}/presence``
    Heartbeat published at :data:`HEARTBEAT_HZ`.  Any subscriber to
    ``strands/*/presence`` (including peers themselves) treats incoming
    payloads as new-or-refreshed peer records.
``strands/{peer_id}/state``
    Numeric state snapshot at :data:`STATE_HZ`.  Camera frames and other
    high-dimensional tensors are filtered out.

Environment variables
---------------------
``STRANDS_MESH``
    Set to ``false`` to disable mesh globally — :func:`init_mesh` returns
    ``None`` and the robot owns no mesh component.

Example
-------
.. code-block:: python

    from strands_robots.mesh import init_mesh

    class MockRobot:
        tool_name_str = "mockbot"

    mesh = init_mesh(MockRobot(), peer_id="mockbot-abcd")
    if mesh is not None:
        print(mesh.alive)        # True if eclipse-zenoh is installed
        print(mesh.peers)        # other peers on the network
        mesh.stop()
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from strands_robots.mesh_audit import log_safety_event
from strands_robots.mesh_session import (
    HEARTBEAT_HZ,
    STATE_HZ,
    current_session,
    get_session,
    prune_peers,
    put,
    release_session,
    update_peer,
)
from strands_robots.mesh_session import (
    get_peers as _session_get_peers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level registry of *local* meshes
# ---------------------------------------------------------------------------

#: All mesh-enabled robots/sims in the current process, keyed by peer_id.
#: Used by ``robot_mesh`` tool (PR6) to find an in-process mesh to send through
#: when the agent issues a command.
_LOCAL_ROBOTS: dict[str, Mesh] = {}
_LOCAL_ROBOTS_LOCK = threading.Lock()


def get_local_robots() -> dict[str, Mesh]:
    """Return a snapshot of in-process mesh-enabled robots.

    Returns a copy — mutating the result does not affect the registry.
    """
    with _LOCAL_ROBOTS_LOCK:
        return dict(_LOCAL_ROBOTS)


# ---------------------------------------------------------------------------
# Mesh — composition component, one per robot/simulation
# ---------------------------------------------------------------------------


class Mesh:
    """Peer-to-peer mesh component embedded in a single Robot or Simulation.

    The :class:`Mesh` is *not* a wrapper around the robot — it is a sibling
    object the robot composes with.  The robot owns the mesh, the mesh keeps a
    weak conceptual back-reference to the robot to read presence and state.

    Construct via :func:`init_mesh`; do not instantiate directly inside
    consumer code.

    Args:
        robot: The owning robot or simulation instance.  Duck-typed: any
            attribute access in :meth:`_build_presence` and :meth:`_read_state`
            is guarded with ``hasattr`` checks.
        peer_id: Stable identifier for this peer on the mesh.
        peer_type: One of ``"robot"``, ``"sim"``, ``"agent"``.

    Thread safety:
        :meth:`start` and :meth:`stop` are protected by ``_lifecycle_lock`` and
        are safe to call concurrently or repeatedly.  The presence and state
        loops run in dedicated daemon threads.
    """

    def __init__(self, robot: Any, peer_id: str, peer_type: str = "robot") -> None:
        self.robot = robot
        self.peer_id = peer_id
        self.peer_type = peer_type

        self._running: bool = False
        self._has_session_ref: bool = False
        self._subs: list[Any] = []
        self._threads: list[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()

        # _subs_lock protects the _subs list and _user_subs dict against
        # concurrent appends from start() / subscribe() / stop().  Without
        # this, two threads calling subscribe() could race the underlying
        # list.append() calls.
        self._subs_lock = threading.Lock()

        # _inbox_lock guards every read/write to self.inbox and its contained
        # buffers.  The subscriber callback writes on the zenoh worker thread;
        # external callers (e.g. the robot_mesh tool) read from the agent
        # thread.  Without this lock the cap-trim logic (`del buf[:N]`) can
        # race with `append` and corrupt the list.
        self._inbox_lock = threading.Lock()

        # _stop_event lets the heartbeat / state loops exit immediately on
        # stop() instead of waiting out the next 0.5s tick.  It is set from
        # stop() and observed by `Event.wait(period)` in each loop.
        self._stop_event = threading.Event()

        # ── RPC correlation state (send/broadcast/tell) ───────────────
        # _pending: turn_id -> Event used by the calling thread to wait
        # _responses: turn_id -> list of response payloads accumulated
        # All access must hold _rpc_lock since the response subscriber
        # thread writes while the calling thread reads/clears.
        self._rpc_lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, list[dict[str, Any]]] = {}

        # ── User-facing subscribe() state ─────────────────────────────
        # inbox[name] -> list[(topic, data)] for buffered subscribers
        # _user_subs[name] -> the zenoh subscription handle
        self.inbox: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._user_subs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Acquire a session reference and start the presence + state loops.

        Idempotent — calling twice is a no-op.  When :mod:`zenoh` is not
        installed (or :func:`get_session` returns ``None`` for any reason) the
        method silently does nothing and the mesh stays in the inactive state.
        """
        with self._lifecycle_lock:
            if self._running:
                return

            session = get_session()
            if session is None:
                logger.debug("[mesh] %s: zenoh unavailable, mesh off", self.peer_id)
                return

            self._has_session_ref = True

            # Subscribers — only presence is needed at the PR2 layer.
            # Command and response subscribers land in PR3; the user-facing
            # subscribe() and stream subscribers land in PR4.
            # Each subscription is declared inside its own try/except so a
            # mid-way failure can be cleanly torn down (Issue #10).  A
            # half-subscribed mesh is worse than no mesh because some keys
            # are observable and others silently aren't.
            declared: list[Any] = []
            try:
                declared.append(session.declare_subscriber("strands/*/presence", self._on_presence))
                declared.append(session.declare_subscriber(f"strands/{self.peer_id}/cmd", self._on_cmd))
                declared.append(session.declare_subscriber("strands/broadcast", self._on_cmd))
                declared.append(
                    session.declare_subscriber(
                        f"strands/{self.peer_id}/response/**",
                        self._on_response,
                    )
                )
            except Exception as exc:  # pragma: no cover — depends on zenoh runtime
                # Tear down whatever we managed to declare so we don't leak
                # subscribers into the shared session.
                for sub in declared:
                    try:
                        sub.undeclare()
                    except Exception as undecl_exc:  # noqa: BLE001
                        logger.debug(
                            "[mesh] %s: cleanup undeclare failed: %s",
                            self.peer_id,
                            undecl_exc,
                        )
                logger.warning("[mesh] %s: failed to declare subscribers: %s", self.peer_id, exc)
                release_session()
                self._has_session_ref = False
                return

            with self._subs_lock:
                self._subs.extend(declared)

            self._running = True
            with _LOCAL_ROBOTS_LOCK:
                _LOCAL_ROBOTS[self.peer_id] = self

            # Start daemon loops *after* registration so any peer that sees our
            # first heartbeat can immediately resolve us.
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name=f"mesh-heartbeat-{self.peer_id}",
                daemon=True,
            )
            state_thread = threading.Thread(
                target=self._state_loop,
                name=f"mesh-state-{self.peer_id}",
                daemon=True,
            )
            self._threads = [heartbeat, state_thread]
            heartbeat.start()
            state_thread.start()

            logger.info("[mesh] %s on mesh (%s)", self.peer_id, self.peer_type)

    def stop(self) -> None:
        """Stop the loops, drop subscribers, release the session reference.

        Idempotent — calling twice is a no-op.  Always pairs with a successful
        :meth:`start`; calling :meth:`stop` on a never-started mesh is a no-op
        and does not release a session reference (because none was acquired).
        """
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            # Wake the heartbeat / state loops so they exit promptly instead
            # of sleeping out the next 0.5s tick (Issue #7).
            self._stop_event.set()

        # Drop from registry first so concurrent peer-listing doesn't see us.
        with _LOCAL_ROBOTS_LOCK:
            _LOCAL_ROBOTS.pop(self.peer_id, None)

        with self._subs_lock:
            subs_to_drop = list(self._subs)
            self._subs.clear()
            self._user_subs.clear()
        with self._inbox_lock:
            self.inbox.clear()

        for sub in subs_to_drop:
            try:
                sub.undeclare()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                # Subscribers belong to a session that may already be closing;
                # log and continue so we always reach release_session().
                logger.debug("[mesh] %s: subscriber undeclare failed: %s", self.peer_id, exc)

        # Wake any callers blocked in send/broadcast so they exit cleanly
        # rather than waiting out a useless timeout against a dead mesh.
        with self._rpc_lock:
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._responses.clear()

        # Daemon threads observe ``_running == False`` on their next tick and
        # exit; we deliberately do not join them to keep stop() non-blocking.

        if self._has_session_ref:
            release_session()
            self._has_session_ref = False

        logger.info("[mesh] %s off mesh", self.peer_id)

    @property
    def alive(self) -> bool:
        """``True`` while the mesh has an active session reference."""
        return self._running

    @property
    def peers(self) -> list[dict[str, Any]]:
        """All peers currently known to this process (excluding self).

        Delegates to :func:`strands_robots.mesh_session.get_peers`.
        """
        return [p for p in _session_get_peers() if p.get("peer_id") != self.peer_id]

    # ------------------------------------------------------------------
    # Presence — outgoing
    # ------------------------------------------------------------------

    def _build_presence(self) -> dict[str, Any]:
        """Build a presence payload by introspecting the owning robot.

        All robot attribute access is wrapped in :class:`AttributeError` /
        broad ``Exception`` guards because the robot is duck-typed and may be
        a partially-initialised object during shutdown.
        """
        r = self.robot
        payload: dict[str, Any] = {
            "robot_id": self.peer_id,
            "robot_type": self.peer_type,
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }

        # Each enrichment block is defensive: if the robot half-implements an
        # interface (e.g. _task_state has no .status), we skip rather than
        # tearing down the heartbeat loop.
        try:
            if hasattr(r, "tool_name_str"):
                payload["tool_name"] = r.tool_name_str
        except Exception:  # noqa: BLE001 — duck-typed introspection
            pass

        try:
            ts = getattr(r, "_task_state", None)
            if ts is not None:
                status = getattr(ts, "status", None)
                payload["task_status"] = getattr(status, "value", status)
                payload["instruction"] = getattr(ts, "instruction", "")
        except Exception:  # noqa: BLE001
            pass

        try:
            inner = getattr(r, "robot", None)
            if inner is not None:
                if hasattr(inner, "is_connected"):
                    payload["connected"] = bool(inner.is_connected)
                if hasattr(inner, "name"):
                    payload["hw"] = inner.name
        except Exception:  # noqa: BLE001
            pass

        try:
            action_features = getattr(r, "_action_features", None)
            if isinstance(action_features, dict):
                payload["action_keys"] = list(action_features.keys())
        except Exception:  # noqa: BLE001
            pass

        try:
            world = getattr(r, "_world", None)
            if world is not None:
                payload["world"] = True
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    payload["sim_robots"] = list(world_robots.keys())
        except Exception:  # noqa: BLE001
            pass

        return payload

    def _heartbeat_loop(self) -> None:
        """Publish presence at :data:`HEARTBEAT_HZ` and prune stale peers.

        Pruning is co-located with the heartbeat tick (rather than in a third
        thread) so a single source of truth drives both publish cadence and
        timeout enforcement.
        """
        period = 1.0 / HEARTBEAT_HZ
        while self._running:
            try:
                put(f"strands/{self.peer_id}/presence", self._build_presence())
                prune_peers()
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                logger.debug("[mesh] %s: heartbeat tick error: %s", self.peer_id, exc)
            # Event.wait returns True when set() is called from stop(), letting
            # us bail out of the sleep immediately instead of waiting out the
            # full period.
            if self._stop_event.wait(period):
                break

    def _on_presence(self, sample: Any) -> None:
        """Subscriber callback for ``strands/*/presence``.

        Decodes the JSON payload and updates the peer registry.  Self-reports
        are ignored.  Any decode error is swallowed silently because mesh
        peers are not authenticated and may publish anything.
        """
        try:
            import json

            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 — untrusted peer payload
            return

        peer_id = data.get("robot_id")
        if not isinstance(peer_id, str) or peer_id == self.peer_id:
            return

        is_new = update_peer(
            peer_id=peer_id,
            peer_type=str(data.get("robot_type", "robot")),
            hostname=str(data.get("hostname", "")),
            caps=data,
        )
        if is_new:
            logger.info("[mesh] new peer: %s (%s)", peer_id, data.get("robot_type", "?"))

    # ------------------------------------------------------------------
    # State — outgoing
    # ------------------------------------------------------------------

    def _state_loop(self) -> None:
        """Publish numeric state at :data:`STATE_HZ`."""
        period = 1.0 / STATE_HZ
        while self._running:
            try:
                state = self._read_state()
                if state:
                    put(f"strands/{self.peer_id}/state", state)
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                logger.debug("[mesh] %s: state tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _read_state(self) -> dict[str, Any] | None:
        """Read a numeric snapshot of the robot's current state.

        Camera frames and any tensor with more than one dimension are
        deliberately excluded — high-rate state is meant for joint positions
        and sim time, not for video.

        Returns ``None`` when no useful state is available so the state loop
        can skip the publish entirely (avoids spamming empty payloads).
        """
        r = self.robot
        snapshot: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

        # Hardware joints (filtered against camera keys).
        try:
            inner = getattr(r, "robot", None)
            if inner is not None and hasattr(inner, "get_observation") and getattr(inner, "is_connected", False):
                obs = inner.get_observation()
                cam_keys = set(getattr(getattr(inner, "config", None), "cameras", {}).keys())
                joints: dict[str, Any] = {}
                for key, value in obs.items():
                    if key in cam_keys:
                        continue
                    shape = getattr(value, "shape", None)
                    if shape is not None and len(shape) > 1:
                        # Exclude image-like tensors even if not in cam_keys.
                        continue
                    if hasattr(value, "tolist"):
                        joints[key] = value.tolist()
                    else:
                        joints[key] = value
                if joints:
                    snapshot["joints"] = joints
        except Exception:  # noqa: BLE001
            pass

        # Task progress.
        try:
            ts = getattr(r, "_task_state", None)
            if ts is not None:
                status = getattr(ts, "status", None)
                snapshot["task"] = {
                    "status": getattr(status, "value", status),
                    "instruction": getattr(ts, "instruction", ""),
                    "steps": getattr(ts, "step_count", 0),
                    "duration": getattr(ts, "duration", 0.0),
                }
        except Exception:  # noqa: BLE001
            pass

        # Simulation clock + active sim robots.
        try:
            world = getattr(r, "_world", None)
            if world is not None:
                world_data = getattr(world, "_data", None)
                if world_data is not None and hasattr(world_data, "time"):
                    snapshot["sim_time"] = float(world_data.time)
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    snapshot["robots"] = {name: {"active": True} for name in world_robots}
        except Exception:  # noqa: BLE001
            pass

        # Only return a snapshot if it has more than the bookkeeping fields.
        return snapshot if len(snapshot) > 2 else None

    # ------------------------------------------------------------------
    # RPC — incoming command dispatch (PR3)
    # ------------------------------------------------------------------

    def _on_cmd(self, sample: Any) -> None:
        """Subscriber callback for ``strands/{peer_id}/cmd`` and
        ``strands/broadcast``.

        Decodes the payload and hands the command off to a daemon thread so
        the subscriber callback returns immediately and the dispatch logic is
        free to call into long-running robot methods.

        Self-loop guard: messages whose ``sender_id`` equals our peer_id are
        ignored (relevant for broadcast).
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 — untrusted peer payload
            return

        if data.get("sender_id") == self.peer_id:
            return

        threading.Thread(
            target=self._exec_cmd,
            args=(data,),
            name=f"mesh-exec-{self.peer_id}",
            daemon=True,
        ).start()

    def _exec_cmd(self, data: dict[str, Any]) -> None:
        """Run dispatch for a received command and publish a response.

        Errors thrown by the robot dispatch are wrapped into a ``type=error``
        response so the caller's :meth:`send` returns a useful payload rather
        than timing out.
        """
        sender = data.get("sender_id", "")
        turn = data.get("turn_id") or uuid.uuid4().hex[:8]
        cmd = data.get("command", data)
        if isinstance(cmd, str):
            # Convenience: raw string command becomes an execute action.
            cmd = {"action": "execute", "instruction": cmd}

        rkey = f"strands/{sender}/response/{turn}" if sender else None

        try:
            result = self._dispatch(cmd)
            if rkey is not None:
                put(
                    rkey,
                    {
                        "type": "response",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "result": result,
                        "timestamp": time.time(),
                    },
                )
        except Exception as exc:  # noqa: BLE001 — never let a thread die
            logger.warning("[mesh] %s: dispatch error: %s", self.peer_id, exc)
            if rkey is not None:
                put(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )

    def _dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Route a parsed command to the owning robot's methods.

        The action vocabulary is intentionally small and stable:
            - ``status``       → robot.get_task_status()
            - ``stop``         → robot.stop_task()
            - ``features``     → robot.get_features()
            - ``state``        → self._read_state()
            - ``execute`` /
              ``start``        → robot._execute_task_sync / start_task
            - ``step``         → robot.step(N) (sim only)
            - ``reset``        → robot.reset() (sim only)

        Unknown actions return ``{"error": "unknown action: ..."}`` rather
        than raising so the wire protocol stays consistent.
        """
        action = cmd.get("action", "status")
        r = self.robot

        if action == "status":
            if hasattr(r, "get_task_status"):
                return dict(r.get_task_status())
            ts = getattr(r, "_task_state", None)
            return {"status": getattr(getattr(ts, "status", None), "value", "unknown")}

        if action == "stop":
            if hasattr(r, "stop_task"):
                return dict(r.stop_task())
            return {"ok": True}

        if action == "features":
            return dict(r.get_features()) if hasattr(r, "get_features") else {}

        if action == "state":
            return self._read_state() or {}

        if action in ("execute", "start"):
            instruction = cmd.get("instruction", "")
            if not instruction:
                return {"error": "instruction required"}
            policy_provider = cmd.get("policy_provider", "mock")
            policy_port = cmd.get("policy_port")
            policy_host = cmd.get("policy_host", "localhost")
            duration = cmd.get("duration", 30.0)
            extra = {
                k: cmd[k]
                for k in (
                    "model_path",
                    "server_address",
                    "policy_type",
                    "pretrained_name_or_path",
                )
                if k in cmd
            }
            if action == "execute" and hasattr(r, "_execute_task_sync"):
                return dict(
                    r._execute_task_sync(instruction, policy_provider, policy_port, policy_host, duration, **extra)
                )
            if action == "start" and hasattr(r, "start_task"):
                return dict(r.start_task(instruction, policy_provider, policy_port, policy_host, duration, **extra))

        if action == "step" and hasattr(r, "step"):
            return dict(r.step(cmd.get("steps", 1)))

        if action == "reset" and hasattr(r, "reset"):
            return dict(r.reset())

        return {"error": f"unknown action: {action}"}

    def _on_response(self, sample: Any) -> None:
        """Subscriber callback for ``strands/{self.peer_id}/response/**``.

        Looks up the matching ``turn_id`` and signals the waiting thread.
        Responses for unknown turn_ids are dropped (the caller already
        timed out).
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 — untrusted peer payload
            return

        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return

        with self._rpc_lock:
            event = self._pending.get(turn)
            if event is None:
                return
            self._responses.setdefault(turn, []).append(data)
        # Set outside the lock so waiting threads can re-acquire if needed.
        event.set()

    # ------------------------------------------------------------------
    # RPC — outgoing (PR3)
    # ------------------------------------------------------------------

    def send(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send a command to a single peer and return the first response.

        Args:
            target: Peer id of the recipient.
            cmd: Command payload — ``{"action": "...", ...}``.
            timeout: Seconds to wait for a response.

        Returns:
            The first response dict, or ``{"status": "timeout"}`` when no
            response arrives in time.
        """
        if not self._running:
            return {"status": "error", "error": "mesh not running"}

        turn = uuid.uuid4().hex[:8]
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []

        msg = {
            "sender_id": self.peer_id,
            "turn_id": turn,
            "command": cmd,
            "timestamp": time.time(),
        }
        try:
            put(f"strands/{target}/cmd", msg)
            event.wait(timeout=timeout)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)

        return resps[0] if resps else {"status": "timeout"}

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        """Broadcast a command to every peer and return all collected responses.

        Args:
            cmd: Command payload — ``{"action": "...", ...}``.
            timeout: Seconds to wait before collecting responses.

        Returns:
            All response dicts received during the timeout window. Empty
            list when no peers respond.
        """
        if not self._running:
            return []

        turn = uuid.uuid4().hex[:8]
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []

        msg = {
            "sender_id": self.peer_id,
            "turn_id": turn,
            "command": cmd,
            "timestamp": time.time(),
        }
        try:
            put("strands/broadcast", msg)
            event.wait(timeout=timeout)
            # Allow late stragglers a brief window after the first response.
            time.sleep(0.3)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
        return resps

    def tell(self, target: str, instruction: str, **kw: Any) -> dict[str, Any]:
        """Shorthand: ask a peer to execute a natural-language instruction.

        Equivalent to ``send(target, {"action": "execute", "instruction": ...})``.
        """
        cmd = {"action": "execute", "instruction": instruction, **kw}
        return self.send(target, cmd)

    # ------------------------------------------------------------------
    # User subscribe / publish_step / on_stream (PR4)
    # ------------------------------------------------------------------

    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, dict[str, Any]], None] | None = None,
        name: str | None = None,
    ) -> str | None:
        """Subscribe to any Zenoh topic and receive parsed JSON dicts.

        Wildcards are supported (e.g. ``"reachy_mini/*"``,
        ``"*/joint_positions"``).  When *callback* is ``None`` messages are
        buffered into ``self.inbox[name]`` so the caller can poll them later.

        Args:
            topic: Zenoh key expression (supports wildcards).
            callback: Called as ``callback(topic, data)`` on each message.
            name: Subscription name used for inbox access and unsubscribe.
                Defaults to *topic*.

        Returns:
            The subscription name (use it with :meth:`unsubscribe`), or
            ``None`` when the mesh is not running.
        """
        if not self._running:
            return None
        # subscribe() piggy-backs on the session reference held by start();
        # using current_session() (no refcount bump) keeps the get/release
        # semantics simple and matches the lifetime contract: the user
        # subscription dies with the mesh.
        session = current_session()
        if session is None:
            return None

        sub_name = name or topic
        with self._inbox_lock:
            self.inbox.setdefault(sub_name, [])

        def handler(sample: Any) -> None:
            try:
                key = str(sample.key_expr)
                raw = sample.payload.to_bytes().decode()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}
                if callback is not None:
                    callback(key, data)
                else:
                    # _inbox_lock prevents the cap-trim logic from racing the
                    # append — without it, two concurrent samples on the same
                    # subscription corrupt the buffer's tail.
                    with self._inbox_lock:
                        buf = self.inbox.setdefault(sub_name, [])
                        buf.append((key, data))
                        if len(buf) > 1000:
                            del buf[: len(buf) - 500]
            except Exception as exc:  # noqa: BLE001 — best-effort callback
                logger.debug("[mesh] %s: subscribe handler error on %s: %s", self.peer_id, topic, exc)

        try:
            sub = session.declare_subscriber(topic, handler)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[mesh] %s: declare_subscriber(%s) failed: %s", self.peer_id, topic, exc)
            return None

        with self._subs_lock:
            self._subs.append(sub)
            self._user_subs[sub_name] = sub
        logger.info("[sub] %s subscribed to: %s", self.peer_id, topic)
        return sub_name

    def unsubscribe(self, name: str) -> None:
        """Unsubscribe from a topic by *name*.

        Idempotent — unknown names are ignored.
        """
        with self._subs_lock:
            sub = self._user_subs.pop(name, None)
            if sub is not None:
                try:
                    self._subs.remove(sub)
                except ValueError:
                    pass
        if sub is None:
            return
        try:
            sub.undeclare()
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.debug("[mesh] %s: subscriber undeclare(%s) failed: %s", self.peer_id, name, exc)
        with self._inbox_lock:
            self.inbox.pop(name, None)

    def publish_step(
        self,
        step: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        instruction: str = "",
        policy: str = "",
    ) -> None:
        """Publish one VLA execution step to the mesh.

        Camera frames and other tensors with more than one dimension are
        filtered out so consumers can subscribe to the stream without paying
        the bandwidth cost of video.
        """
        if not self._running:
            return

        obs_numeric: dict[str, Any] = {}
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 1:
                continue  # skip images / tensors
            if hasattr(value, "tolist"):
                obs_numeric[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str)):
                obs_numeric[key] = value
            elif isinstance(value, (list, tuple)) and len(value) < 100:
                obs_numeric[key] = list(value)

        act_numeric: dict[str, Any] = {}
        for key, value in action.items():
            if hasattr(value, "tolist"):
                act_numeric[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str, list, tuple)):
                act_numeric[key] = value if not isinstance(value, tuple) else list(value)

        put(
            f"strands/{self.peer_id}/stream",
            {
                "peer_id": self.peer_id,
                "step": step,
                "t": time.time(),
                "instruction": instruction,
                "policy": policy,
                "observation": obs_numeric,
                "action": act_numeric,
            },
        )

    def on_stream(
        self,
        peer_id: str,
        callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str | None:
        """Subscribe to another peer's VLA execution stream.

        Convenience wrapper around :meth:`subscribe` that subscribes to
        ``strands/{peer_id}/stream`` under the name ``stream:{peer_id}``.
        """
        return self.subscribe(f"strands/{peer_id}/stream", callback, name=f"stream:{peer_id}")

    # ------------------------------------------------------------------
    # Safety — emergency stop with audit log (PR5)
    # ------------------------------------------------------------------

    def emergency_stop(self) -> list[dict[str, Any]]:
        """Broadcast a stop command to every peer and audit the event.

        The audit log lives at ``~/.strands_robots/mesh_audit.jsonl`` (override
        with ``STRANDS_MESH_AUDIT_DIR``) with mode ``0o600``.
        """
        responses = self.broadcast({"action": "stop"}, timeout=3.0)
        try:
            log_safety_event(
                event_type="emergency_stop",
                peer_id=self.peer_id,
                payload={
                    "sender_id": self.peer_id,
                    "responses_received": len(responses),
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit log must never break stop
            logger.warning("[mesh] audit log write failed: %s", exc)
        return responses


# ---------------------------------------------------------------------------
# init_mesh — the only public constructor
# ---------------------------------------------------------------------------


def init_mesh(
    robot: Any,
    peer_id: str | None = None,
    peer_type: str = "robot",
    mesh: bool = True,
) -> Mesh | None:
    """Construct and start a :class:`Mesh` for *robot*.

    This is the single entry point used by ``Robot.__init__`` and
    ``Simulation.__init__`` (PR6 wires those callers).  Consumers should treat
    the returned ``Mesh`` as *opaque* — store it on ``self.mesh`` and call
    ``self.mesh.stop()`` during cleanup.

    Args:
        robot: The owning instance.
        peer_id: Stable peer identifier.  When ``None``, derives one from
            ``robot.tool_name_str`` plus a 4-character UUID suffix.
        peer_type: ``"robot"``, ``"sim"``, or ``"agent"``.
        mesh: When ``False``, returns ``None`` immediately and acquires no
            session reference.

    Returns:
        A started :class:`Mesh` instance, or ``None`` when mesh is disabled
        either by the *mesh* argument or by ``STRANDS_MESH=false`` in the
        environment.

    Notes:
        Per the project's environment-variable conventions, the kill switch
        ``STRANDS_MESH`` accepts case-insensitive ``"false"`` / ``"true"``
        with surrounding whitespace ignored.
    """
    env = os.getenv("STRANDS_MESH", "true").strip().lower()
    if env == "false":
        mesh = False

    if not mesh:
        return None

    if peer_id is None:
        base = getattr(robot, "tool_name_str", None) or "robot"
        # 8 hex chars = 32 bits of entropy — avoids collisions when many
        # peers join the mesh at once. Larger than the 4-char prefix used by
        # the dev branch so tests stay forward-compatible.
        peer_id = f"{base}-{uuid.uuid4().hex[:8]}"

    instance = Mesh(robot, peer_id=peer_id, peer_type=peer_type)
    instance.start()
    return instance


__all__ = [
    "Mesh",
    "get_local_robots",
    "init_mesh",
]
