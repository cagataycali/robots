"""Observer mesh peer for the dashboard.

The dashboard is not a robot, but it still wants first-class mesh access:
discover peers, subscribe to presence/state/camera streams, and dispatch
commands (teleop, execute, e-stop). Rather than open a second raw Zenoh
session the way the original prototype did, we construct a real
:class:`~strands_robots.mesh.Mesh` around a tiny stub "robot". This gives the
dashboard the exact same wire path as any other peer — same transport, same
ACL, same command validation and audit — for free.

The stub robot exposes no actuators and no observations: ``_dispatch`` on the
mesh will only ever answer ``status`` for it (everything else returns an
"unknown action" / no-capability response), so the dashboard peer cannot be
driven by another peer. It is observe-and-command-out only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _DashboardStubRobot:
    """Minimal robot-shaped object so :class:`Mesh` can wrap the dashboard.

    Mesh introspects its ``robot`` via ``getattr(..., default)`` for every
    capability (joints, cameras, world, teleop, ...). A bare object therefore
    advertises *no* capabilities, which is exactly what we want: the dashboard
    publishes presence but answers no actuating commands.
    """

    tool_name_str = "dashboard"


class MeshObserver:
    """Wraps a dashboard-type :class:`Mesh` and bridges traffic to a sink.

    Parameters
    ----------
    on_event:
        Called as ``on_event(kind, payload)`` for every mesh event the
        dashboard cares about. ``kind`` is one of ``presence``, ``state``,
        ``camera``, ``stream``, ``event``. Runs on a Zenoh callback thread,
        so the sink must be cheap and thread-safe (the server hands these to
        an asyncio queue).
    peer_id:
        Mesh peer id for the dashboard. Defaults to a random ``dashboard-*``.
    """

    # Topics the dashboard subscribes to. Presence is already handled by the
    # mesh's own peer registry, but we also watch it here to push join/leave
    # events to the browser in real time.
    _WILDCARD_TOPICS = (
        ("state", "strands/*/state"),
        ("stream", "strands/*/stream"),
        ("event", "strands/*/event"),
    )

    def __init__(
        self,
        on_event: Callable[[str, dict[str, Any]], None],
        peer_id: str | None = None,
    ) -> None:
        self._on_event = on_event
        self._peer_id = peer_id
        self._mesh: Any = None
        self._camera_subs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._known_peers: dict[str, float] = {}
        self._teleop_seq: int = 0
        self._teleop_device: str = "dashboard"

    # -- lifecycle ------------------------------------------------------

    def start(self) -> bool:
        """Join the mesh. Returns True if the mesh came up alive."""
        from strands_robots.mesh import init_mesh

        try:
            self._mesh = init_mesh(
                _DashboardStubRobot(),
                peer_id=self._peer_id,
                peer_type="dashboard",
            )
        except Exception as exc:  # noqa: BLE001 — surface as a dead observer
            logger.error("dashboard mesh init failed: %s", exc)
            self._mesh = None
            return False

        if self._mesh is None or not self._mesh.alive:
            logger.warning(
                "dashboard mesh is not alive (STRANDS_MESH=false or transport down)"
            )
            return False

        self._peer_id = self._mesh.peer_id

        # Wildcard subscriptions for state/stream/event.
        for kind, topic in self._WILDCARD_TOPICS:
            self._mesh.subscribe(
                topic,
                callback=self._make_handler(kind),
                name=f"dash-{kind}",
            )

        # Camera topics are per-peer (strands/{peer}/camera/{cam}); a single
        # wildcard strands/*/camera/* catches them all.
        self._mesh.subscribe(
            "strands/*/camera/*",
            callback=self._make_handler("camera"),
            name="dash-camera",
        )

        # 3D scene streams (sim peers): baked geometry (once) + live geom
        # poses (per frame). Wildcards catch every sim peer on the mesh.
        self._mesh.subscribe(
            "strands/*/scene/geom",
            callback=self._make_handler("scene_geom"),
            name="dash-scene-geom",
        )
        self._mesh.subscribe(
            "strands/*/scene/pose",
            callback=self._make_handler("scene_pose"),
            name="dash-scene-pose",
        )

        logger.info("dashboard observer joined mesh as %s", self._peer_id)
        return True

    def stop(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("dashboard mesh stop error: %s", exc)
            self._mesh = None

    # -- properties -----------------------------------------------------

    @property
    def peer_id(self) -> str | None:
        return self._peer_id

    @property
    def alive(self) -> bool:
        return self._mesh is not None and self._mesh.alive

    def peers(self) -> list[dict[str, Any]]:
        """Snapshot of all known peers from the mesh registry."""
        from strands_robots.mesh import get_peers

        try:
            return get_peers()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_peers failed: %s", exc)
            return []

    # -- command-out path ----------------------------------------------

    def send(self, target: str, cmd: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        """Send an RPC command to one peer and return its response."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(target, cmd, timeout=timeout)

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        """Broadcast a command to all peers and return all responses."""
        if self._mesh is None:
            return []
        return self._mesh.broadcast(cmd, timeout=timeout)

    def emergency_stop(self) -> list[dict[str, Any]]:
        """Broadcast a stop to every peer. Audited by the mesh layer."""
        return self.broadcast({"action": "stop"}, timeout=3.0)

    # -- teleop (dashboard as leader) -----------------------------------

    def start_teleop(self, target: str, device_name: str | None = None) -> dict[str, Any]:
        """Ask *target* to follow this dashboard's input stream.

        The dashboard publishes input frames on
        ``strands/{dashboard_peer}/input/{device}``; the target subscribes
        via its ``teleop_receive`` action. After this returns success, call
        :meth:`teleop_frame` to stream joint targets.
        """
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        device = device_name or self._teleop_device
        self._teleop_seq = 0
        return self._mesh.send(
            target,
            {
                "action": "teleop_receive",
                "source_peer_id": self._mesh.peer_id,
                "device_name": device,
            },
            timeout=10.0,
        )

    def teleop_frame(self, action: dict[str, float], device_name: str | None = None,
                     events: dict[str, Any] | None = None) -> None:
        """Publish a single teleop input frame on the dashboard's input topic.

        ``action`` is a flat ``{joint_name: float}`` dict. The receiving peer
        validates + clamps each frame (see ``validate_input_frame``) before
        applying via ``send_action``, so an out-of-range value is dropped at
        the follower, not actuated.
        """
        if self._mesh is None:
            return
        device = device_name or self._teleop_device
        topic = f"strands/{self._mesh.peer_id}/input/{device}"
        payload = {
            "peer_id": self._mesh.peer_id,
            "device": device,
            "method": "keyboard",
            "t": time.time(),
            "seq": self._teleop_seq,
            "action": {k: float(v) for k, v in action.items()},
            "events": events,
        }
        self._teleop_seq += 1
        self._mesh.publish(topic, payload)

    def calibrate(self, target: str, step: str, timeout: float = 15.0) -> dict[str, Any]:
        """Drive the web calibration state machine on *target*.

        ``step`` is one of begin/home/record/finish/cancel/status. The longer
        default timeout accommodates set_half_turn_homings which does a bus
        round-trip per motor.
        """
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(target, {"action": "calibrate", "step": step}, timeout=timeout)

    def stop_teleop(self, target: str, device_name: str | None = None) -> dict[str, Any]:
        """Tell *target* to stop following this dashboard's input stream."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        device = device_name or self._teleop_device
        return self._mesh.send(
            target,
            {"action": "teleop_stop", "device_name": device},
            timeout=10.0,
        )

    # -- policies + recording (dashboard introspection / data collection) ---

    def list_policies(self, target: str, timeout: float = 10.0) -> dict[str, Any]:
        """Ask *target* for its available policy providers + running set."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(target, {"action": "list_policies"}, timeout=timeout)

    def list_robots(self, target: str, timeout: float = 10.0) -> dict[str, Any]:
        """Ask *target* for the robot names it hosts (sim scene robots)."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(target, {"action": "list_robots"}, timeout=timeout)

    def record_start(self, target: str, repo_id: str, task: str = "",
                     fps: int = 30, overwrite: bool = True,
                     timeout: float = 20.0) -> dict[str, Any]:
        """Start LeRobot dataset recording on a sim *target*."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(
            target,
            {"action": "record_start", "repo_id": repo_id, "task": task,
             "fps": fps, "overwrite": overwrite},
            timeout=timeout,
        )

    def record_stop(self, target: str, timeout: float = 30.0) -> dict[str, Any]:
        """Stop dataset recording on a sim *target* and finalise the episode."""
        if self._mesh is None:
            return {"status": "error", "error": "mesh not running"}
        return self._mesh.send(target, {"action": "record_stop"}, timeout=timeout)

    # -- internals ------------------------------------------------------

    def _make_handler(self, kind: str) -> Callable[[str, dict[str, Any]], None]:
        def handler(key: str, data: dict[str, Any]) -> None:
            try:
                payload = dict(data)
                payload["_topic"] = key
                self._on_event(kind, payload)
            except Exception as exc:  # noqa: BLE001 — never kill the sub thread
                logger.debug("observer handler error (%s): %s", kind, exc)

        return handler
