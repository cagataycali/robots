"""Core Mesh class — lifecycle, presence, state, cameras, RPC, and subscriptions.

This is the primary component that a Robot or Simulation composes with.
Extended sensor loops (pose, IMU, health, etc.) are provided by
:class:`~strands_robots.mesh.sensors.SensorLoopsMixin`.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from strands_robots.mesh import security as _security
from strands_robots.mesh.sensors import SensorLoopsMixin
from strands_robots.mesh.session import (
    CAMERA_HZ,
    HEARTBEAT_HZ,
    STATE_HZ,
    current_session,
    get_session,
    prune_peers,
    put,
    release_session,
    update_peer,
)
from strands_robots.mesh.session import (
    get_peers as _session_get_peers,
)

logger = logging.getLogger(__name__)


# Module-level registry of local meshes
_LOCAL_ROBOTS: dict[str, Mesh] = {}
_LOCAL_ROBOTS_LOCK = threading.Lock()


def get_local_robots() -> dict[str, Mesh]:
    """Return a snapshot of in-process mesh-enabled robots."""
    with _LOCAL_ROBOTS_LOCK:
        return dict(_LOCAL_ROBOTS)


class Mesh(SensorLoopsMixin):
    """Peer-to-peer mesh component embedded in a single Robot or Simulation.

    Lifecycle: construct via :func:`init_mesh`, call :meth:`stop` during cleanup.

    Thread safety:
        :meth:`start` and :meth:`stop` are protected by ``_lifecycle_lock``.
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
        self._subs_lock = threading.Lock()
        self._inbox_lock = threading.Lock()
        self._stop_event = threading.Event()

        # RPC correlation state
        self._rpc_lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, list[dict[str, Any]]] = {}

        # User subscribe state
        self.inbox: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._user_subs: dict[str, Any] = {}

        # Emergency-stop lockout flag. While this Event is set, every
        # action other than ``status`` and ``resume`` is refused (see
        # :meth:`_dispatch`). The flag is cleared by :meth:`_resume_lockout`,
        # which requires the operator-supplied override code.
        self._estop_lockout = threading.Event()
        self._last_estop_ts: float = 0.0

    def __repr__(self) -> str:
        state = "alive" if self._running else "stopped"
        return f"Mesh(peer_id={self.peer_id!r}, type={self.peer_type!r}, {state})"

    # Lifecycle
    def start(self) -> None:
        """Acquire a Zenoh session and start all publishing loops."""
        with self._lifecycle_lock:
            if self._running:
                return

            session = get_session()
            if session is None:
                logger.debug("[mesh] %s: zenoh unavailable, mesh off", self.peer_id)
                return

            self._has_session_ref = True

            declared: list[Any] = []
            try:
                declared.append(session.declare_subscriber("strands/*/presence", self._on_presence))
                declared.append(session.declare_subscriber(f"strands/{self.peer_id}/cmd", self._on_cmd))
                declared.append(session.declare_subscriber("strands/broadcast", self._on_cmd))
                declared.append(session.declare_subscriber(f"strands/{self.peer_id}/response/**", self._on_response))
                # Fleet-wide e-stop: any peer broadcasting on safety/estop or
                # safety/resume engages / clears the lockout on every other
                # peer too. Without these subscribers the lockout would only
                # protect the issuing process, leaving receivers willing to
                # accept the next command after they've stopped the current
                # task.
                declared.append(session.declare_subscriber("strands/safety/estop", self._on_safety_estop))
                declared.append(session.declare_subscriber("strands/safety/resume", self._on_safety_resume))
            except Exception as exc:
                for sub in declared:
                    try:
                        sub.undeclare()
                    except Exception:
                        pass
                logger.warning("[mesh] %s: failed to declare subscribers: %s", self.peer_id, exc)
                release_session()
                self._has_session_ref = False
                return

            with self._subs_lock:
                self._subs.extend(declared)

            self._running = True
            with _LOCAL_ROBOTS_LOCK:
                _LOCAL_ROBOTS[self.peer_id] = self

            # Core loops
            heartbeat = threading.Thread(
                target=self._heartbeat_loop, name=f"mesh-heartbeat-{self.peer_id}", daemon=True
            )
            state_thread = threading.Thread(target=self._state_loop, name=f"mesh-state-{self.peer_id}", daemon=True)
            self._threads = [heartbeat, state_thread]
            heartbeat.start()
            state_thread.start()

            # Optional camera loop
            camera_hz = self._resolve_camera_hz()
            if camera_hz > 0:
                cam_thread = threading.Thread(
                    target=self._camera_loop,
                    args=(camera_hz,),
                    name=f"mesh-camera-{self.peer_id}",
                    daemon=True,
                )
                self._threads.append(cam_thread)
                cam_thread.start()
                logger.info("[mesh] %s camera stream enabled @ %.1f Hz", self.peer_id, camera_hz)

            # Extended sensor loops (from SensorLoopsMixin)
            extended_loops = [
                ("pose", self._pose_loop),
                ("health", self._health_loop),
                ("imu", self._imu_loop),
                ("odom", self._odom_loop),
                ("lidar", self._lidar_loop),
                ("hand", self._hand_loop),
                ("map-info", self._map_info_loop),
            ]
            for loop_name, loop_fn in extended_loops:
                t = threading.Thread(target=loop_fn, name=f"mesh-{loop_name}-{self.peer_id}", daemon=True)
                self._threads.append(t)
                t.start()

            logger.info("[mesh] %s on mesh (%s)", self.peer_id, self.peer_type)

    def stop(self) -> None:
        """Stop all loops and release the session reference."""
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

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
            except Exception:
                pass

        with self._rpc_lock:
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._responses.clear()

        if self._has_session_ref:
            release_session()
            self._has_session_ref = False

        logger.info("[mesh] %s off mesh", self.peer_id)

    @property
    def alive(self) -> bool:
        return self._running

    @property
    def peers(self) -> list[dict[str, Any]]:
        return [p for p in _session_get_peers() if p.get("peer_id") != self.peer_id]

    # Presence — outgoing
    def _build_presence(self) -> dict[str, Any]:
        r = self.robot
        payload: dict[str, Any] = {
            "robot_id": self.peer_id,
            "robot_type": self.peer_type,
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }

        try:
            if hasattr(r, "tool_name_str"):
                payload["tool_name"] = r.tool_name_str
        except Exception:
            pass

        try:
            ts = getattr(r, "_task_state", None)
            if ts is not None:
                status = getattr(ts, "status", None)
                payload["task_status"] = getattr(status, "value", status)
                payload["instruction"] = getattr(ts, "instruction", "")
        except Exception:
            pass

        try:
            inner = getattr(r, "robot", None)
            if inner is not None:
                if hasattr(inner, "is_connected"):
                    payload["connected"] = bool(inner.is_connected)
                if hasattr(inner, "name"):
                    payload["hw"] = inner.name
                cam_cfg = getattr(getattr(inner, "config", None), "cameras", None)
                if isinstance(cam_cfg, dict) and cam_cfg:
                    payload["cameras"] = list(cam_cfg.keys())
                input_pubs = getattr(r, "_input_publishers", None)
                if isinstance(input_pubs, dict) and input_pubs:
                    payload["inputs"] = [
                        {"device": name, "method": pub.method, "hz": pub.hz}
                        for name, pub in input_pubs.items()
                        if pub._running
                    ]
        except Exception:
            pass

        try:
            action_features = getattr(r, "_action_features", None)
            if isinstance(action_features, dict):
                payload["action_keys"] = list(action_features.keys())
        except Exception:
            pass

        try:
            world = getattr(r, "_world", None)
            if world is not None:
                payload["world"] = True
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    payload["sim_robots"] = list(world_robots.keys())
        except Exception:
            pass

        # Advertise available extended topics
        available_topics: list[str] = []
        try:
            if (
                getattr(r, "_pose", None) is not None
                or getattr(r, "_slam_pose", None) is not None
                or getattr(r, "_odom_pose", None) is not None
            ):
                available_topics.append("pose")
            if getattr(r, "_imu", None) is not None:
                available_topics.append("imu")
            if getattr(r, "_odom", None) is not None:
                available_topics.append("odom")
            if getattr(r, "_lidar_summary", None) is not None or getattr(r, "_lidar_state", None) is not None:
                available_topics.append("lidar")
            if getattr(r, "_battery", None) is not None:
                available_topics.append("health")
            if getattr(r, "_hands", None) is not None:
                available_topics.append("hand")
            if getattr(r, "_map_info", None) is not None:
                available_topics.append("map")
        except Exception:
            pass
        if "health" not in available_topics:
            available_topics.append("health")
        if available_topics:
            payload["topics"] = available_topics

        return payload

    def _heartbeat_loop(self) -> None:
        period = 1.0 / HEARTBEAT_HZ
        while self._running:
            try:
                self._put_signed(f"strands/{self.peer_id}/presence", self._build_presence())
                prune_peers()
            except Exception as exc:
                logger.debug("[mesh] %s: heartbeat tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _on_presence(self, sample: Any) -> None:
        try:
            raw = sample.payload.to_bytes().decode()
            envelope = json.loads(raw)
        except Exception:
            return
        # Verify presence signature. In strict mode the verifier rejects
        # unsigned presence; in permissive mode legacy un-enveloped messages
        # pass through unchanged for back-compat with un-upgraded peers.
        try:
            data = _security.verify_envelope(envelope, scope=self.peer_id)
        except _security.AuthenticationError as exc:
            logger.debug("[mesh] %s: rejected unauthenticated presence: %s", self.peer_id, exc)
            return
        if not isinstance(data, dict):
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

    # State — outgoing
    def _state_loop(self) -> None:
        period = 1.0 / STATE_HZ
        while self._running:
            try:
                state = self._read_state()
                if state:
                    self._put_signed(f"strands/{self.peer_id}/state", state)
            except Exception as exc:
                logger.debug("[mesh] %s: state tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _read_state(self) -> dict[str, Any] | None:
        r = self.robot
        snapshot: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

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
                        continue
                    if hasattr(value, "tolist"):
                        joints[key] = value.tolist()
                    else:
                        joints[key] = value
                if joints:
                    snapshot["joints"] = joints
        except Exception:
            pass

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
        except Exception:
            pass

        try:
            world = getattr(r, "_world", None)
            if world is not None:
                world_data = getattr(world, "_data", None)
                if world_data is not None and hasattr(world_data, "time"):
                    snapshot["sim_time"] = float(world_data.time)
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    snapshot["robots"] = {name: {"active": True} for name in world_robots}
        except Exception:
            pass

        return snapshot if len(snapshot) > 2 else None

    # Cameras — outgoing (opt-in)
    def _resolve_camera_hz(self) -> float:
        env = os.getenv("STRANDS_MESH_CAMERA_HZ")
        if env is None or env.strip() == "":
            hz = CAMERA_HZ
        else:
            try:
                hz = float(env)
            except ValueError:
                logger.warning("STRANDS_MESH_CAMERA_HZ=%r invalid; camera loop disabled", env)
                return 0.0
        return hz if hz > 0 else 0.0

    def _camera_loop(self, hz: float) -> None:
        period = 1.0 / hz
        while self._running:
            try:
                self._publish_cameras_once()
            except Exception as exc:
                logger.debug("[mesh] %s: camera tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _publish_cameras_once(self) -> None:
        # Privacy kill switch. Operators on sensitive deployments set
        # STRANDS_MESH_CAMERA_DISABLED=true to short-circuit the camera
        # loop entirely — no frames built, no envelopes signed, nothing
        # published.
        if os.getenv("STRANDS_MESH_CAMERA_DISABLED", "").strip().lower() == "true":
            return
        r = self.robot
        inner = getattr(r, "robot", None)
        if inner is None or not getattr(inner, "is_connected", False):
            return
        cam_cfg = getattr(getattr(inner, "config", None), "cameras", None)
        if not isinstance(cam_cfg, dict) or not cam_cfg:
            return

        obs = None
        try:
            obs = inner.get_observation()
        except Exception:
            pass

        if obs is None:
            cameras_dict = getattr(inner, "cameras", None)
            if not isinstance(cameras_dict, dict) or not cameras_dict:
                return
            obs = {}
            for cam_name, cam_obj in cameras_dict.items():
                try:
                    if hasattr(cam_obj, "async_read"):
                        obs[cam_name] = cam_obj.async_read()
                    elif hasattr(cam_obj, "read"):
                        obs[cam_name] = cam_obj.read()
                except Exception:
                    pass
            if not obs:
                return

        try:
            import cv2

            have_cv2 = True
        except Exception:
            have_cv2 = False

        for cam_name in cam_cfg:
            try:
                frame = obs.get(cam_name)
                if frame is None:
                    continue
                shape = getattr(frame, "shape", None)
                if shape is None or len(shape) < 2:
                    continue
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                if hasattr(frame, "astype"):
                    import numpy as np

                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)

                if have_cv2:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
                    encoding = "jpeg"
                else:
                    encoded = base64.b64encode(bytes(frame)).decode("ascii")
                    encoding = "raw"

                self._put_signed(
                    f"strands/{self.peer_id}/camera/{cam_name}",
                    {
                        "peer_id": self.peer_id,
                        "cam": cam_name,
                        "t": time.time(),
                        "shape": list(shape),
                        "dtype": "uint8",
                        "encoding": encoding,
                        "data": encoded,
                    },
                )
            except Exception as exc:
                logger.debug("[mesh] %s: camera %s publish failed: %s", self.peer_id, cam_name, exc)

    # RPC — incoming
    def _on_cmd(self, sample: Any) -> None:
        try:
            raw = sample.payload.to_bytes().decode()
            envelope = json.loads(raw)
        except Exception:
            return
        # Authenticate the envelope: HMAC signature + freshness window +
        # replay nonce. Unauthenticated commands are dropped at the first
        # step here — _exec_cmd never runs for them. Scope = self.peer_id
        # so other Mesh peers in the same process maintain independent
        # nonce caches and a broadcast does not trip "replay" on the
        # second receiver.
        try:
            data = _security.verify_envelope(envelope, scope=self.peer_id)
        except _security.AuthenticationError as exc:
            logger.warning("[mesh] %s: rejected unauthenticated cmd: %s", self.peer_id, exc)
            return
        if not isinstance(data, dict):
            return
        sender_id = data.get("sender_id", "")
        if sender_id == self.peer_id:
            return
        # Per-sender token-bucket rate limit. Without this, a flood of
        # commands from one peer could spawn arbitrarily many exec threads
        # and exhaust the robot's resources.
        if not _security.consume_peer_token(sender_id or self.peer_id):
            logger.warning(
                "[mesh] %s: per-sender rate limit exceeded for sender=%s",
                self.peer_id,
                sender_id or "<anon>",
            )
            try:
                from strands_robots.mesh.audit import log_safety_event

                log_safety_event(
                    "rate_limit_drop",
                    self.peer_id,
                    {"sender": sender_id, "topic": "cmd"},
                )
            except Exception as audit_exc:
                # Audit is best-effort — never let a missing/broken log
                # path break the cmd handler. Logged at DEBUG so it shows
                # up under verbose logging without spamming production.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return
        threading.Thread(target=self._exec_cmd, args=(data,), name=f"mesh-exec-{self.peer_id}", daemon=True).start()

    def _exec_cmd(self, data: dict[str, Any]) -> None:
        sender = data.get("sender_id", "")
        turn = data.get("turn_id") or uuid.uuid4().hex[:8]
        cmd = data.get("command", data)
        if isinstance(cmd, str):
            cmd = {"action": "execute", "instruction": cmd}
        rkey = f"strands/{sender}/response/{turn}" if sender else None

        # Validate the command shape against the action allowlist + per-action
        # schema (instruction length, duration bounds, policy_host allowlist, ...).
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: rejected invalid cmd from %s: %s", self.peer_id, sender, exc)
            if rkey is not None:
                self._put_signed(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": f"validation: {exc}",
                        "timestamp": time.time(),
                    },
                )
            try:
                from strands_robots.mesh.audit import log_safety_event

                log_safety_event(
                    "command_rejected",
                    self.peer_id,
                    {
                        "sender": sender,
                        "reason": str(exc),
                        "action": cmd.get("action") if isinstance(cmd, dict) else None,
                    },
                )
            except Exception as audit_exc:
                # Audit is best-effort — never let a missing/broken log
                # path swallow the validation rejection itself.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return

        try:
            result = self._dispatch(cmd)
            if rkey is not None:
                self._put_signed(
                    rkey,
                    {
                        "type": "response",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "result": result,
                        "timestamp": time.time(),
                    },
                )
        except _security.LockoutError as exc:
            # Lockout is the most operationally interesting rejection — emit
            # a structured error on the response topic and audit it.
            logger.warning("[mesh] %s: rejected during lockout from %s", self.peer_id, sender)
            if rkey is not None:
                self._put_signed(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
            try:
                from strands_robots.mesh.audit import log_safety_event

                log_safety_event(
                    "command_rejected_lockout",
                    self.peer_id,
                    {"sender": sender, "action": cmd.get("action") if isinstance(cmd, dict) else None},
                )
            except Exception as audit_exc:
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return
        except Exception as exc:
            logger.warning("[mesh] %s: dispatch error: %s", self.peer_id, exc)
            if rkey is not None:
                self._put_signed(
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
        action = cmd.get("action", "status")
        r = self.robot

        # While the emergency-stop lockout is engaged, only ``status`` and
        # ``resume`` are permitted. Raise so _exec_cmd handles the rejection
        # symmetrically with ValidationError — emitting type="error" on the
        # response topic and recording an audit entry. The wire response is
        # intentionally generic so a remote caller cannot use it to map the
        # lockout window.
        if self._estop_lockout.is_set() and action not in ("status", "resume"):
            raise _security.LockoutError("command rejected")

        if action == "resume":
            return self._resume_lockout(cmd.get("override_code", ""))

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
                for k in ("model_path", "server_address", "policy_type", "pretrained_name_or_path")
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
        if action == "teleop_status":
            if hasattr(r, "get_teleop_status"):
                return dict(r.get_teleop_status())
            return {"inputs": [], "publishers": {}, "receivers": {}}
        if action == "teleop_receive":
            source = cmd.get("source_peer_id", "")
            dev = cmd.get("device_name", "leader")
            if not source:
                return {"error": "source_peer_id required"}
            if hasattr(r, "start_teleop_receive"):
                return dict(r.start_teleop_receive(source, dev))
            return {"error": "robot does not support teleop_receive"}
        if action == "teleop_stop":
            dev = cmd.get("device_name")
            if hasattr(r, "stop_teleop"):
                return dict(r.stop_teleop(dev))
            return {"error": "robot does not support stop_teleop"}
        return {"error": f"unknown action: {action}"}

    def _on_response(self, sample: Any) -> None:
        try:
            raw = sample.payload.to_bytes().decode()
            envelope = json.loads(raw)
        except Exception:
            return
        try:
            data = _security.verify_envelope(envelope, scope=self.peer_id)
        except _security.AuthenticationError as exc:
            logger.debug("[mesh] %s: rejected unauthenticated response: %s", self.peer_id, exc)
            return
        if not isinstance(data, dict):
            return
        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return
        with self._rpc_lock:
            event = self._pending.get(turn)
            if event is None:
                return
            self._responses.setdefault(turn, []).append(data)
        event.set()

    # Safety — inbound estop / resume
    def _on_safety_estop(self, sample: Any) -> None:
        """Engage the local emergency-stop lockout in response to a fleet-
        wide ``strands/safety/estop`` broadcast.

        The verifier rejects unsigned envelopes in strict mode, so an
        attacker without the PSK cannot trigger a fleet lockout. The
        sender is still allowed to be ourselves — engaging the flag
        twice is a no-op.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            envelope = json.loads(raw)
        except Exception:
            return
        try:
            data = _security.verify_envelope(envelope, scope=self.peer_id)
        except _security.AuthenticationError as exc:
            logger.warning("[mesh] %s: rejected unauthenticated estop: %s", self.peer_id, exc)
            return
        if not isinstance(data, dict):
            return
        if not self._estop_lockout.is_set():
            self._estop_lockout.set()
            self._last_estop_ts = time.time()
            sender = data.get("peer_id", "<remote>")
            logger.critical("[safety] %s: lockout engaged via remote estop from %s", self.peer_id, sender)

    def _on_safety_resume(self, sample: Any) -> None:
        """Clear the local lockout in response to ``strands/safety/resume``.

        The publisher of the resume event already validated the operator
        override code locally (see :meth:`_resume_lockout`); we trust the
        signed envelope and follow.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            envelope = json.loads(raw)
        except Exception:
            return
        try:
            data = _security.verify_envelope(envelope, scope=self.peer_id)
        except _security.AuthenticationError as exc:
            logger.warning("[mesh] %s: rejected unauthenticated resume: %s", self.peer_id, exc)
            return
        if not isinstance(data, dict):
            return
        if self._estop_lockout.is_set():
            self._estop_lockout.clear()
            sender = data.get("peer_id", "<remote>")
            logger.warning("[safety] %s: lockout cleared via remote resume from %s", self.peer_id, sender)

    # RPC — outgoing
    def send(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send a command to a single peer and return the first response."""
        if not self._running:
            return {"status": "error", "error": "mesh not running"}
        turn = uuid.uuid4().hex[:8]
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self._put_signed(f"strands/{target}/cmd", msg)
            event.wait(timeout=timeout)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
        return resps[0] if resps else {"status": "timeout"}

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        """Broadcast a command to every peer and return all responses."""
        if not self._running:
            return []
        turn = uuid.uuid4().hex[:8]
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self._put_signed("strands/broadcast", msg)
            event.wait(timeout=timeout)
            time.sleep(0.3)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
        return resps

    def tell(self, target: str, instruction: str, **kw: Any) -> dict[str, Any]:
        """Shorthand: ask a peer to execute a natural-language instruction."""
        return self.send(target, {"action": "execute", "instruction": instruction, **kw})

    # Subscribe / publish_step / on_stream
    def subscribe(
        self, topic: str, callback: Callable[[str, dict[str, Any]], None] | None = None, name: str | None = None
    ) -> str | None:
        """Subscribe to any Zenoh topic and receive parsed JSON dicts."""
        if not self._running:
            return None
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
                    with self._inbox_lock:
                        buf = self.inbox.setdefault(sub_name, [])
                        buf.append((key, data))
                        if len(buf) > 1000:
                            del buf[: len(buf) - 500]
            except Exception as exc:
                logger.debug("[mesh] %s: subscribe handler error on %s: %s", self.peer_id, topic, exc)

        try:
            sub = session.declare_subscriber(topic, handler)
        except Exception as exc:
            logger.warning("[mesh] %s: declare_subscriber(%s) failed: %s", self.peer_id, topic, exc)
            return None

        with self._subs_lock:
            self._subs.append(sub)
            self._user_subs[sub_name] = sub
        logger.info("[sub] %s subscribed to: %s", self.peer_id, topic)
        return sub_name

    def unsubscribe(self, name: str) -> None:
        """Unsubscribe from a topic by name."""
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
        except Exception:
            pass
        with self._inbox_lock:
            self.inbox.pop(name, None)

    def publish_step(
        self, step: int, observation: dict[str, Any], action: dict[str, Any], instruction: str = "", policy: str = ""
    ) -> None:
        """Publish one VLA execution step to the mesh."""
        if not self._running:
            return
        obs_numeric: dict[str, Any] = {}
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 1:
                continue
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

        self._put_signed(
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

    def on_stream(self, peer_id: str, callback: Callable[[str, dict[str, Any]], None] | None = None) -> str | None:
        """Subscribe to another peer's VLA execution stream."""
        return self.subscribe(f"strands/{peer_id}/stream", callback, name=f"stream:{peer_id}")

    # Safety — emergency stop
    def emergency_stop(self) -> list[dict[str, Any]]:
        """Broadcast a stop command to every peer and engage the local lockout.

        After this call the local mesh refuses every non-status, non-resume
        action until :meth:`_resume_lockout` is invoked with the operator
        override code (``STRANDS_MESH_OVERRIDE_CODE``). The event is also
        published on ``strands/safety/estop`` and recorded in the audit log
        (see :func:`strands_robots.mesh.audit.log_safety_event`).

        Returns the list of responses received from peers within the broadcast
        timeout — useful for telemetry (which peers acknowledged before the
        stop fanned out).
        """
        self._estop_lockout.set()
        self._last_estop_ts = time.time()
        responses = self.broadcast({"action": "stop"}, timeout=3.0)
        self._put_signed(
            "strands/safety/estop",
            {
                "peer_id": self.peer_id,
                "t": self._last_estop_ts,
                "responses_received": len(responses),
                "lockout_engaged": True,
            },
        )
        self.publish_safety_event(
            event_type="emergency_stop",
            severity="critical",
            payload={
                "sender_id": self.peer_id,
                "responses_received": len(responses),
                "lockout_engaged": True,
            },
        )
        logger.critical("[safety] %s: EMERGENCY STOP engaged — lockout active", self.peer_id)
        return responses

    def _resume_lockout(self, override_code: str) -> dict[str, Any]:
        """Clear the emergency-stop lockout if *override_code* matches.

        Compared in constant time against ``STRANDS_MESH_OVERRIDE_CODE``.
        Mismatch returns an error and emits a ``resume_denied`` safety
        event; success emits ``resume_ok`` and publishes
        ``strands/safety/resume`` so peers can refresh their view.

        Returns a status dict::

            {"status": "ok",    "lockout_elapsed_s": <seconds>}    # success
            {"status": "noop",  "lockout": False}                  # not engaged
            {"status": "error", "error": "<reason>"}               # rejected

        Every attempt — successful or not — is recorded in the audit log
        through :meth:`publish_safety_event`.
        """
        if not self._estop_lockout.is_set():
            return {"status": "noop", "lockout": False}

        expected = os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip()
        provided = (override_code or "").strip()
        if not expected:
            self.publish_safety_event(
                event_type="resume_denied",
                severity="warning",
                payload={"sender_id": self.peer_id, "reason": "STRANDS_MESH_OVERRIDE_CODE not configured"},
            )
            return {"status": "error", "error": "override code not configured"}
        # Constant-time compare so an attacker probing the override code
        # cannot use response-time variation to learn it byte-by-byte.
        if not hmac.compare_digest(expected.encode(), provided.encode()):
            self.publish_safety_event(
                event_type="resume_denied",
                severity="warning",
                payload={"sender_id": self.peer_id, "reason": "bad override code"},
            )
            return {"status": "error", "error": "invalid override code"}

        elapsed = time.time() - self._last_estop_ts
        self._estop_lockout.clear()
        self.publish_safety_event(
            event_type="resume_ok",
            severity="info",
            payload={"sender_id": self.peer_id, "lockout_elapsed_s": elapsed},
        )
        self._put_signed(
            "strands/safety/resume",
            {"peer_id": self.peer_id, "t": time.time(), "lockout_elapsed_s": elapsed},
        )
        logger.warning("[safety] %s: resume after %.1fs lockout", self.peer_id, elapsed)
        return {"status": "ok", "lockout_elapsed_s": elapsed}

    def _put_signed(self, key: str, payload: dict[str, Any]) -> None:
        """Sign *payload* into an envelope and publish it on *key*.

        Centralises the signing call so every outgoing mesh message goes
        through one code path.

        Failure mode: when a PSK is configured (or strict-auth is set) and
        :func:`sign_envelope` raises, the message is **dropped** rather than
        sent unsigned. In permissive mode (no PSK and strict-auth off) the
        helper falls back to publishing the raw payload — there is nothing
        to downgrade in that case anyway.
        """
        try:
            envelope = _security.sign_envelope(payload)
        except Exception as exc:
            # If a PSK was configured we must NOT silently downgrade. A
            # caller that can trigger sign_envelope() to raise — corrupted
            # key material, OS entropy exhaustion, etc. — would otherwise
            # cause every subsequent message to leave unsigned, which
            # defeats authentication entirely.
            if _security.psk_configured() or _security.auth_required():
                logger.error(
                    "[mesh] %s: sign_envelope failed for %s under signed mode — DROPPING: %s",
                    self.peer_id,
                    key,
                    exc,
                )
                return
            logger.warning(
                "[mesh] %s: sign_envelope failed for %s (permissive mode, sending unsigned): %s",
                self.peer_id,
                key,
                exc,
            )
            envelope = payload
        put(key, envelope)


# init_mesh — the only public constructor
def init_mesh(
    robot: Any,
    peer_id: str | None = None,
    peer_type: str = "robot",
    mesh: bool = True,
) -> Mesh | None:
    """Construct and start a Mesh for the given robot.

    Returns None when mesh is disabled (STRANDS_MESH=false or mesh=False).
    """
    env = os.getenv("STRANDS_MESH", "true").strip().lower()
    if env == "false":
        mesh = False
    if not mesh:
        return None

    if peer_id is None:
        base = getattr(robot, "tool_name_str", None) or "robot"
        peer_id = f"{base}-{uuid.uuid4().hex[:8]}"

    # Validate peer_id — reject reserved names and MQTT-unsafe characters.
    _RESERVED_PEER_IDS = {"broadcast", "safety"}
    _PEER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,127}$")
    if peer_id in _RESERVED_PEER_IDS:
        raise ValueError(
            f"peer_id={peer_id!r} is reserved for system use. Reserved names: {sorted(_RESERVED_PEER_IDS)}"
        )
    if not _PEER_ID_PATTERN.match(peer_id):
        raise ValueError(
            f"peer_id={peer_id!r} contains invalid characters. "
            "Must match [a-zA-Z0-9][a-zA-Z0-9._-]{{0,127}} "
            "(no /, +, # — these break MQTT topic structure and AWS Thing-name rules)."
        )

    instance = Mesh(robot, peer_id=peer_id, peer_type=peer_type)
    instance.start()

    # Auto-wire IoT enrichments when the active transport supports them.
    # Both calls are no-ops when STRANDS_MESH_BACKEND=zenoh (the default),
    # so this is purely additive — Zenoh-LAN behaviour is unchanged.
    if instance.alive:
        try:
            from strands_robots.mesh.iot import (
                enable_camera_offload_for_mesh,
                enable_shadow_for_mesh,
            )

            enable_shadow_for_mesh(instance)
            enable_camera_offload_for_mesh(instance)
        except Exception as exc:  # noqa: BLE001 — IoT enrichment is best-effort
            logger.debug("[mesh] IoT enrichment failed (continuing): %s", exc)

    return instance
