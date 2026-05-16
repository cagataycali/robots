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

import logging
import os
import socket
import threading
import time
import uuid
from typing import Any

from strands_robots.mesh_session import (
    HEARTBEAT_HZ,
    STATE_HZ,
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
            try:
                self._subs.append(session.declare_subscriber("strands/*/presence", self._on_presence))
            except Exception as exc:  # pragma: no cover — depends on zenoh runtime
                # If we cannot subscribe to presence we cannot discover peers,
                # so abort cleanly rather than running half-broken.
                logger.warning("[mesh] %s: failed to declare presence subscriber: %s", self.peer_id, exc)
                release_session()
                self._has_session_ref = False
                return

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

        # Drop from registry first so concurrent peer-listing doesn't see us.
        with _LOCAL_ROBOTS_LOCK:
            _LOCAL_ROBOTS.pop(self.peer_id, None)

        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                # Subscribers belong to a session that may already be closing;
                # log and continue so we always reach release_session().
                logger.debug("[mesh] %s: subscriber undeclare failed: %s", self.peer_id, exc)
        self._subs.clear()

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
            time.sleep(period)

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
            time.sleep(period)

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
        peer_id = f"{base}-{uuid.uuid4().hex[:4]}"

    instance = Mesh(robot, peer_id=peer_id, peer_type=peer_type)
    instance.start()
    return instance


__all__ = [
    "Mesh",
    "get_local_robots",
    "init_mesh",
]
