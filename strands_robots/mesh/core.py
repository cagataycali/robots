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
from strands_robots.mesh.audit import log_safety_event
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


#: Sentinel stored in :attr:`Mesh._expected_responders` for
#: broadcast turn_ids. Distinct from any real peer_id (no peer_id
#: contains a NUL byte).
BROADCAST_RESPONDER: str = "<broadcast>\x00"


def _parse_positive_float_env(name: str, default: str, *, minimum: float = 0.0) -> float:
    """Parse a positive-float env var, falling back to default on bad input.

    Catches the case where an operator sets ``STRANDS_MESH_RESUME_FRESHNESS_S=abc``
    or a negative value. The module would otherwise fail to import with an opaque
    ``ValueError`` (found by running the module under bad env locally; see
    ``test_resume_env_validation.py``).
    """
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r (not a float); falling back to default %r.",
            name,
            raw,
            default,
        )
        return float(default)
    if value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %s); falling back to default %r.",
            name,
            value,
            minimum,
            default,
        )
        return float(default)
    return value


def _parse_positive_int_env(name: str, default: str, *, minimum: int = 1) -> int:
    """Parse a positive-int env var, falling back to default on bad input.

    Companion to :func:`_parse_positive_float_env` for cache-size knobs.
    Rejects zero / negative because ``deque(maxlen=0)`` would silently disable
    the replay cache and ``maxlen=-1`` would raise at runtime.
    """
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r (not an int); falling back to default %r.",
            name,
            raw,
            default,
        )
        return int(default)
    if value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %s); falling back to default %r.",
            name,
            value,
            minimum,
            default,
        )
        return int(default)
    return value


#: Default resume-envelope freshness window. Envelopes whose t field
#: is older than this are rejected as potential replays. Operators
#: on drifty NTP can extend via STRANDS_MESH_RESUME_FRESHNESS_S
#: (sane bound: keep < 600). Bad input falls back to 60.
#:
#: F9-B (PR #195 review): the module-level value below is captured
#: once at import time for backward compatibility with code/tests
#: that read these as constants. The runtime hot paths
#: (:meth:`Mesh._on_safety_estop`, :meth:`Mesh._on_safety_resume`)
#: now call :func:`_resume_freshness_window_s` etc. on every call so
#: an operator setting STRANDS_MESH_RESUME_* AFTER importing the
#: module sees the new value without a process restart. The README
#: contract ("operator-tunable env vars") now actually holds.
RESUME_FRESHNESS_WINDOW_S: float = _parse_positive_float_env("STRANDS_MESH_RESUME_FRESHNESS_S", "60")

#: Forward-skew tolerance on the envelope t field. See
#: :data:`RESUME_FRESHNESS_WINDOW_S` for the lazy-resolution note.
RESUME_FORWARD_SKEW_S: float = _parse_positive_float_env("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")

#: Maximum entries in the per-receiver resume replay cache.
#: See :data:`RESUME_FRESHNESS_WINDOW_S` for lazy-resolution note.
RESUME_REPLAY_CACHE_MAX: int = _parse_positive_int_env("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4096")


def _resume_freshness_window_s() -> float:
    """Lazy resolver for ``STRANDS_MESH_RESUME_FRESHNESS_S``.

    F9-B: re-reads the env var on every call so operator-set values
    take effect without a process restart. Cheap (one ``os.getenv``
    + a regex parse via ``_parse_positive_float_env``) and called
    only on safety-handler entry, which is bounded by the transport
    rate cap.
    """
    return _parse_positive_float_env("STRANDS_MESH_RESUME_FRESHNESS_S", "60")


def _resume_forward_skew_s() -> float:
    """Lazy resolver for ``STRANDS_MESH_RESUME_FORWARD_SKEW_S``."""
    return _parse_positive_float_env("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")


def _resume_replay_cache_max() -> int:
    """Lazy resolver for ``STRANDS_MESH_RESUME_REPLAY_CACHE_MAX``."""
    return _parse_positive_int_env("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4096")


def _evict_replay_cache[K](
    cache: dict[K, float],
    *,
    max_size: int,
    ttl_s: float,
    now_mono: float,
) -> None:
    """Bound *cache* to ``max_size`` entries in-place.

    Eviction strategy (single-pass, no-op when the cache is below the cap):

    1. Drop entries whose stored monotonic timestamp is older than
       ``now_mono - ttl_s`` (TTL purge -- aligned with the safety-replay
       freshness window so an entry that no longer protects against any
       in-window replay is free to evict).
    2. If the cache is still over budget after the TTL purge (cap is full
       of in-window entries -- active flood), drop the oldest 20 percent
       by stored timestamp.

    Single source of truth for both ``_estop_replay_cache`` and
    ``_resume_replay_cache`` eviction. Callers own insertion and the
    per-cache lock; this helper is pure bookkeeping.
    """
    if len(cache) < max_size:
        return
    cutoff = now_mono - ttl_s
    stale = [k for k, ts in cache.items() if ts < cutoff]
    for k in stale:
        cache.pop(k, None)
    if len(cache) >= max_size:
        ordered = sorted(cache.items(), key=lambda kv: kv[1])
        drop = max(1, len(ordered) // 5)
        for k, _ in ordered[:drop]:
            cache.pop(k, None)


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

        # RPC correlation state.
        #
        # _expected_responders maps turn_id -> the peer_id we expect to
        # answer (set by send() at point-to-point), or the sentinel
        # ``BROADCAST_RESPONDER`` if the turn_id was created by
        # broadcast() and we accept responses from any peer. Phase-4 /
        # D1: this is what _on_response uses to reject a forged
        # response from a peer that wasn't the original target.
        self._rpc_lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, list[dict[str, Any]]] = {}
        self._expected_responders: dict[str, str] = {}

        # User subscribe state
        self.inbox: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._user_subs: dict[str, Any] = {}

        # Emergency-stop lockout flag. While this Event is set, every
        # action other than ``status`` and ``resume`` is refused (see
        # :meth:`_dispatch`). The flag is cleared by :meth:`_resume_lockout`,
        # which requires the operator-supplied override code.
        self._estop_lockout = threading.Event()
        self._last_estop_ts: float = 0.0
        # _on_safety_resume must defend
        # against replay of a previously-observed override-proof envelope.
        # The receiver caches (proof_nonce, issuer_peer_id) tuples it has
        # already accepted and refuses duplicates within a bounded window.
        # Combined with the freshness check on the envelope ``t`` field
        # this closes the recorded-and-replayed-resume surface even when
        # an attacker has live ACL access on safety/**.
        self._resume_replay_cache: dict[tuple[str, str], float] = {}
        self._resume_replay_lock = threading.Lock()
        # R9: estop replay defense -- mirror of resume cache, keyed on
        # (issuer_peer_id, envelope_t). Closes the captured-estop-replay DoS
        # surface that previously let any peer with live ACL access to
        # safety/** replay a captured envelope indefinitely. Reuses
        # _resume_freshness_window_s() / _resume_forward_skew_s() /
        # _resume_replay_cache_max() (the safety-replay defenses are
        # symmetric in shape; sharing the bounds keeps env-var surface
        # minimal).
        # F9-A (PR #195 review): cache shape is
        # ``dict[float_t, (issuer_id, mono_ts)]``. The float key
        # preserves the R20 peer_id-permutation defence (attacker
        # cannot mint novel keys by varying untrusted JSON peer_id);
        # the value carries issuer attribution AND the eviction
        # timestamp. Per-issuer counts are derived from cache
        # contents on demand -- no separate dict that can drift
        # after eviction (the F7-E flaw).
        self._estop_replay_cache: dict[float, tuple[str, float]] = {}
        self._estop_replay_lock = threading.Lock()

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
            # Warn when running with permissive default ACL under mtls --
            # any cert-holder can publish on **/cmd and **/safety/**. This is
            # documented but easy to miss; log loudly at session-open so it
            # surfaces during incident review (reviewer ask, R19).
            try:
                from strands_robots.mesh._acl_config import is_default_acl_in_use
                from strands_robots.mesh._zenoh_config import resolve_auth_mode

                if resolve_auth_mode() == "mtls" and is_default_acl_in_use():
                    # F9-E (PR #195 review): promote the
                    # ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`` opt-in from
                    # "silences the ERROR" to "required to start at all"
                    # under ``mtls + permissive default ACL``. Per the
                    # threat-vector table, this combination turns a
                    # single CA-signed compromise into a fleet-wide
                    # command-issuance pivot -- the worst tier in the
                    # document. AGENTS.md > Safety Defaults > "Sim-by-
                    # default" calls for the inverse: refuse to start
                    # under the dangerous combination unless the
                    # operator has explicitly opted in.
                    #
                    # Three deployment shapes:
                    #
                    # * Production: ``STRANDS_MESH_ACL_FILE`` set ->
                    #   no warning, no opt-in needed.
                    # * Dev/lab opt-in: ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1``
                    #   -> INFO log, mesh starts.
                    # * Default (no opt-in, no ACL file): mesh REFUSES
                    #   to start with a clear error explaining the two
                    #   ways forward.
                    accept = os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "").strip().lower()
                    if accept in ("1", "true", "yes"):
                        logger.info(
                            "[mesh] %s: permissive default ACL active under mtls "
                            "(operator opted in via STRANDS_MESH_ACCEPT_PERMISSIVE_ACL).",
                            self.peer_id,
                        )
                    else:
                        # F11-B (PR #195 review): refuse-and-return,
                        # not refuse-and-raise. Pre-F11 raised a
                        # RuntimeError from Mesh.start which is called
                        # transitively from Robot()/Simulation()
                        # constructors via init_mesh -- that crashed
                        # the first-run UX of `pip install
                        # strands-robots[mesh]`. The safety property
                        # we actually want is "no permissive ACL is
                        # on the wire" -- preserved by returning
                        # early before session acquisition.
                        # Construction succeeds; mesh.alive stays
                        # False; operator sees the loud ERROR.
                        msg = (
                            "PERMISSIVE DEFAULT ACL ACTIVE UNDER MTLS -- any "
                            "CA-signed peer would be able to publish on "
                            "**/cmd and **/safety/**. This combination "
                            "turns a single compromised cert into a "
                            "fleet-wide command pivot. Mesh NOT STARTED. "
                            "Either:\n"
                            "  1. Set STRANDS_MESH_ACL_FILE to a literal-CN "
                            "     ACL file enumerating role separation "
                            "     (production posture; see "
                            "     examples/mesh_acl_example.json5).\n"
                            "  2. Set STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 "
                            "     to acknowledge the dev/lab posture "
                            "     explicitly (single-tenant only).\n"
                            "  3. Set STRANDS_MESH=false to disable the "
                            "     mesh entirely."
                        )
                        logger.error("[mesh] %s: %s", self.peer_id, msg)
                        # Return early WITHOUT acquiring a session.
                        # ``self._running`` stays False; ``self.alive``
                        # returns False; no wire activity occurs.
                        # Pin: tests/mesh/test_default_acl_warning.py
                        # ::test_mtls_plus_default_acl_does_not_start
                        return
            except (ImportError, ValueError) as warn_exc:
                # Narrowed per AGENTS.md > "Exception Clauses Must Be Narrow":
                # ImportError covers the lazy ``from . import _acl_config``;
                # ValueError covers ``resolve_auth_mode`` raising on bad env.
                # An unexpected exception type is a programmer bug we want
                # to see in logs, not a silently-swallowed warning loss.
                logger.warning(
                    "[mesh] %s: ACL posture check raised %s: %s -- permissive-ACL warning may be missing; investigate",
                    self.peer_id,
                    type(warn_exc).__name__,
                    warn_exc,
                )

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
                self.publish(f"strands/{self.peer_id}/presence", self._build_presence())
                prune_peers()
            except Exception as exc:
                logger.debug("[mesh] %s: heartbeat tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _on_presence(self, sample: Any) -> None:
        """Handle a peer's presence broadcast.

        Identity, fleet membership, and replay protection are enforced
        at the Zenoh transport: a sample reaching this callback has
        already cleared mTLS handshake + ACL, so its peer-id is
        cryptographically bound to the cert CN. We only parse the
        payload, update our peer registry, and log a debug line for
        first-sighting.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except Exception:
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
                    self.publish(f"strands/{self.peer_id}/state", state)
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
        # loop entirely -- no frames built, no envelopes signed, nothing
        # published.
        # Lenient bool parsing matches the rest of the env-var surface
        # (STRANDS_MESH_MULTICAST, STRANDS_MESH_I_KNOW_THIS_IS_INSECURE).
        # Operators using ``=1`` / ``=yes`` / ``=on`` get the same
        # behaviour as ``=true``; bad values fail-loud rather than
        # silently re-enabling camera publishing on a privacy flag.
        from strands_robots.mesh._zenoh_config import _bool_env as _zc_bool_env

        if _zc_bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False):
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

                self.publish(
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
        """Handle an inbound command sample.

        The Zenoh transport has already enforced:

        * mTLS peer identity (the sender's cert CN is bound to the link).
        * ACL -- **when the operator supplies ``STRANDS_MESH_ACL_FILE``
          with role separation, only peers in the ``operator_peer``
          subject can publish on ``cmd`` / ``broadcast`` topics**. The
          default ``default_acl()`` is permissive (any CA-signed peer
          may publish/subscribe on any key) -- see CHANGELOG.md Section 8.
        * Per-key-expression frequency cap (``downsampling`` block) --
          floods are dropped pre-deserialise.
        * Per-message size cap (``low_pass_filter`` block) -- jumbo
          frames are dropped pre-deserialise.

        We only have to parse the payload and dispatch.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        sender_id = data.get("sender_id", "")
        if sender_id == self.peer_id:
            return
        threading.Thread(
            target=self._exec_cmd,
            args=(data,),
            name=f"mesh-exec-{self.peer_id}",
            daemon=True,
        ).start()

    def _exec_cmd(self, data: dict[str, Any]) -> None:
        sender = data.get("sender_id", "")
        # R5-1: full 128-bit fallback. Pre-fix, an inbound command without
        # turn_id triggered a 32-bit hex which was birthday-colliding under
        # heavy concurrent load and cheap to predict for an attacker who
        # could observe the response topic. D1 closed the outbound side;
        # this closes the symmetric receive-side surface.
        turn = data.get("turn_id") or uuid.uuid4().hex
        cmd = data.get("command", data)
        # R24-B: Reject non-dict commands at the wire boundary. A bare-string
        # coercion here would bypass validate_command's dict-shape contract --
        # any peer that survives mTLS+ACL could drive the robot at the mock
        # policy with arbitrary text simply by publishing "hello" instead of
        # {"action":"execute",...}. Outgoing send/broadcast/tell still accept
        # the ergonomic dict-or-string forms because tell() wraps internally.
        # See PR #195 thread PRRT_kwDORUMiZs6EUu8S.
        if not isinstance(cmd, dict):
            logger.warning(
                "[mesh] %s: rejected non-dict cmd from %s (type=%s)",
                self.peer_id,
                sender,
                type(cmd).__name__,
            )
            return
        rkey = f"strands/{sender}/response/{turn}" if sender else None

        # Validate the command shape against the action allowlist + per-action
        # schema (instruction length, duration bounds, policy_host allowlist, ...).
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: rejected invalid cmd from %s: %s", self.peer_id, sender, exc)
            if rkey is not None:
                self.publish(
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
                log_safety_event(
                    "command_rejected",
                    self.peer_id,
                    {
                        "sender": sender,
                        "reason": str(exc),
                        "action": cmd.get("action") if isinstance(cmd, dict) else None,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                # F7-F (PR #195 review): narrow per AGENTS.md > Review
                # Learnings (#86). ``log_safety_event`` raises TypeError
                # / ValueError on payload shape, OSError on disk failure;
                # the audit best-effort contract means we drop those, but
                # an unexpected RuntimeError from a future audit-module
                # refactor should NOT be silently swallowed.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return

        try:
            result = self._dispatch(cmd)
            if rkey is not None:
                self.publish(
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
            # Lockout is the most operationally interesting rejection -- emit
            # a structured error on the response topic and audit it.
            logger.warning("[mesh] %s: rejected during lockout from %s", self.peer_id, sender)
            if rkey is not None:
                self.publish(
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
                log_safety_event(
                    "command_rejected_lockout",
                    self.peer_id,
                    {"sender": sender, "action": cmd.get("action") if isinstance(cmd, dict) else None},
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                # F7-F (PR #195 review): narrow per AGENTS.md > "Exception
                # Clauses Must Be Narrow". Same tuple as the symmetric
                # narrowing at the ValidationError audit path above.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return
        except (
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            OSError,
            TypeError,
        ) as exc:
            # F8-C (PR #195 review): narrowed from bare ``except Exception``
            # per AGENTS.md > Review Learnings (#86). The original goal
            # ("any unhandled exception in a robot adapter would crash
            # the dispatch thread and silently kill the mesh") is
            # achievable with a narrow tuple: this catches every
            # realistic adapter failure (LeRobot raising RuntimeError,
            # GR00T raising ValueError on bad inputs, type mismatches,
            # missing keys, OSError from device I/O) but lets
            # ``MemoryError``, ``SystemExit``, ``KeyboardInterrupt``,
            # and any future programmer-error type that doesn't fit
            # this list propagate to the test harness / supervisor.
            #
            # The "static dispatch error string on the wire" rationale
            # below stays the same -- regardless of catch width, we
            # never leak internal exception detail to a remote caller.
            # The structured ValidationError / LockoutError paths above
            # remain the preferred channel for client-distinguishable
            # rejections.
            logger.warning(
                "[mesh] %s: dispatch error from %s: %s",
                self.peer_id,
                sender,
                exc,
                exc_info=True,
            )
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": "dispatch error",
                        "timestamp": time.time(),
                    },
                )

    def _dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = cmd.get("action", "status")
        r = self.robot

        # While the emergency-stop lockout is engaged, only ``status`` and
        # ``resume`` are permitted. Raise so _exec_cmd handles the rejection
        # symmetrically with ValidationError -- emitting type="error" on the
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
        """Inbound response handler.

        Identity, fleet membership, and topic ACL have already been
        enforced at the Zenoh transport. We additionally apply a
        point-to-point scope check: a response is accepted only if its
        ``responder_id`` matches the expected target recorded in
        :attr:`_expected_responders` by :meth:`send`. Broadcast turns
        use the ``BROADCAST_RESPONDER`` sentinel and accept any
        responder_id -- that is the broadcast contract.

        Without the responder-id check, an ACL-authorised peer that
        observes a turn_id (a fellow operator) could publish a response
        on someone else's pending turn and have the sender accept its
        ``result`` instead of the legitimate target's. The transport
        ACL prevents an attacker from joining at all; this check
        prevents lateral mischief between authorised peers.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return
        responder = data.get("responder_id")
        with self._rpc_lock:
            event = self._pending.get(turn)
            if event is None:
                return
            expected = self._expected_responders.get(turn)
            # Strict scoping for point-to-point sends. Broadcast accepts any.
            if expected is not None and expected != BROADCAST_RESPONDER and responder != expected:
                # R8-3: structured forensic event via the audit log
                # (``response_hijack_rejected``) plus a WARNING line is
                # the operator-and-forensic channel; an earlier draft
                # also raised-and-caught a typed exception around the
                # same code, but with no real consumer it was YAGNI
                # scaffolding and got removed.
                logger.warning(
                    "[mesh] %s: dropped response on turn %s -- "
                    "responder_id=%r does not match expected target %r "
                    "(possible response hijack)",
                    self.peer_id,
                    turn[:12],
                    responder,
                    expected,
                )
                try:
                    log_safety_event(
                        "response_hijack_rejected",
                        self.peer_id,
                        {
                            "turn_prefix": turn[:12],
                            "responder_id": responder,
                            "expected": expected,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # F9-C (PR #195 review): narrow per AGENTS.md > Review
                    # Learnings (#86). Same tuple as the other audit-publish
                    # wrappers in this file. Audit best-effort still holds;
                    # MemoryError / RuntimeError / future programmer errors
                    # propagate to the test harness instead of being
                    # silently swallowed at DEBUG.
                    logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
                return
            self._responses.setdefault(turn, []).append(data)
        event.set()

    # Safety -- inbound estop / resume
    def _on_safety_estop(self, sample: Any) -> None:
        """Engage the local emergency-stop lockout in response to a fleet-
        wide ``strands/safety/estop`` broadcast.

        Wire authentication (mTLS + ACL) admits this handler. **When the
        operator supplies an ``STRANDS_MESH_ACL_FILE`` with role
        separation (template at ``examples/mesh_acl_example.json5``),
        only peers in the ``operator_peer`` subject can publish on
        ``safety/**``.** The default ACL shipped by ``default_acl()`` is
        permissive (CHANGELOG.md Section 8 -- "any CA-signed peer may
        publish/subscribe on any key"), so any cert-holding peer can
        originate an estop on out-of-the-box deployments.

        R9 defense-in-depth -- captured-envelope replay protection
        (PR #195 review feedback): even with an unrestricted ACL, a
        replay of a captured ``safety/estop`` envelope cannot keep the
        fleet locked indefinitely.  Mirrors :meth:`_on_safety_resume`:

        1. Freshness window (``_resume_freshness_window_s()``) -- envelopes
           older than the window are rejected.
        2. Forward-skew bound (``_resume_forward_skew_s()``) -- envelopes
           timestamped beyond the tolerance in the future are rejected
           (defeats clock-rollback attacks against the freshness check).
        3. Per-receiver replay cache keyed on ``float(envelope_t)``
           (R20) -- bounded LRU at ``_resume_replay_cache_max()`` entries.
           Keyed on ``t`` alone (NOT ``(issuer_id, t)``) so an attacker
           who captures one envelope cannot replay it by varying the
           payload ``peer_id`` field, which is untrusted (comes from the
           JSON body, not the TLS cert CN).

        E-stop without an envelope ``t`` OR without a valid string
        ``peer_id`` is rejected as malformed (the canonical
        :meth:`emergency_stop` issuer always sets both).
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        # R9: freshness + replay defenses. An estop envelope without
        # ``t`` is not from a canonical issuer -- reject (also closes
        # the trivial replay surface where an attacker strips ``t`` to
        # bypass the freshness check).
        envelope_t = data.get("t")
        now = time.time()
        if not isinstance(envelope_t, (int, float)):
            logger.warning(
                "[safety] %s: refusing remote estop -- envelope missing/invalid ``t``",
                self.peer_id,
            )
            return
        if envelope_t > now + _resume_forward_skew_s():
            logger.warning(
                "[safety] %s: refusing remote estop -- ``t``=%s in future (forward_skew_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                _resume_forward_skew_s(),
                now,
            )
            return
        if (now - envelope_t) > _resume_freshness_window_s():
            logger.warning(
                "[safety] %s: refusing remote estop -- ``t``=%s too old (freshness_window_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                _resume_freshness_window_s(),
                now,
            )
            return

        # R20: reject envelopes with missing/empty ``peer_id`` outright
        # rather than coalescing to ``<unknown>``. The canonical
        # :meth:`emergency_stop` issuer always sets ``peer_id``; a
        # malformed envelope is either a programming bug or an attacker
        # probing the cache. Coalescing to a shared bucket let one
        # attacker poison the slot for legitimate operators.
        issuer_id = data.get("peer_id")
        if not isinstance(issuer_id, str) or not issuer_id:
            logger.warning(
                "[safety] %s: refusing remote estop -- envelope missing/invalid ``peer_id``",
                self.peer_id,
            )
            return

        # R20: cache key is keyed on ``float(envelope_t)`` ALONE -- not
        # ``(issuer_id, t)``. The previous (issuer, t) key let an
        # attacker who captured one valid envelope replay it
        # indefinitely by varying the payload ``peer_id`` (which is
        # untrusted -- it comes from the JSON body, not the TLS cert
        # CN). Keying on the wall-clock ``t`` alone closes that
        # peer_id-permutation surface; the only way to mint a new key
        # is to advance the timestamp, which is bounded by the
        # freshness window above. F7-E adds a per-issuer slot cap
        # below to bound the denial-of-estop surface where one
        # attacker pre-publishes ``t = now + skew - eps`` to occupy
        # cache slots that legitimate same-float-tick estops would
        # collide on. See PR #195 R20 thread on core.py:1026 + F7-E
        # thread on the post-R20 denial-of-estop surface.
        cache_key = float(envelope_t)
        # F7-E per-issuer fairness bound: one issuer may occupy at
        # most ``per_issuer_cap`` slots so a single attacker cannot
        # fill the global cache. Default cap is _resume_replay_cache_max()
        # / 4 -- four legitimate operators always have working slots.
        per_issuer_cap = max(1, _resume_replay_cache_max() // 4)
        # R19: cache TTL bookkeeping uses time.monotonic() so an NTP step
        # backward cannot leave entries un-evictable and a step forward
        # cannot age fresh entries out early. Envelope freshness still
        # uses time.time() above (it must compare against the issuer's
        # wall-clock). See PR #195 audit B5.
        now_mono = time.monotonic()
        with self._estop_replay_lock:
            if cache_key in self._estop_replay_cache:
                # R22-B: If lockout is active and within 0.2s, this is corroboration
                # (two distinct operators, same t) not a replay attack.
                if self._estop_lockout.is_set() and (time.time() - self._last_estop_ts) < 0.2:
                    try:
                        self.publish_safety_event(
                            event_type="estop_corroborated",
                            severity="info",
                            payload={"issuer": issuer_id, "issuer_t": envelope_t},
                        )
                    except (TypeError, ValueError, OSError) as audit_exc:
                        # Narrow per AGENTS.md > "Exception Clauses Must Be Narrow".
                        # publish_safety_event raises only TypeError/ValueError
                        # on payload shape and OSError on disk failure; everything
                        # else is a programmer bug worth seeing.
                        logger.debug(
                            "[mesh] %s: estop_corroborated audit publish failed: %s",
                            self.peer_id,
                            audit_exc,
                        )
                    return
                # Original replay rejection
                logger.warning(
                    "[safety] %s: REJECTED remote estop -- replay of (issuer=%s, t=%s) already accepted",
                    self.peer_id,
                    issuer_id,
                    envelope_t,
                )
                try:
                    self.publish_safety_event(
                        event_type="estop_replay_rejected",
                        severity="warning",
                        payload={"issuer": issuer_id, "issuer_t": envelope_t},
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # Audit publish is best-effort and must never block the
                    # safety path; narrow the catch tuple so a future
                    # programmer error surfaces in tests rather than
                    # being swallowed at DEBUG.
                    logger.debug(
                        "[mesh] %s: estop_replay_rejected audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
                return
            # F9-A: evict using the tuple-valued cache. We extract a
            # mono_ts view, run the standard eviction, then re-key
            # the surviving entries from the original cache.
            ts_view: dict[float, float] = {k: v[1] for k, v in self._estop_replay_cache.items()}
            _evict_replay_cache(
                ts_view,
                max_size=_resume_replay_cache_max(),
                ttl_s=_resume_freshness_window_s(),
                now_mono=now_mono,
            )
            # Apply the eviction back to the real cache.
            for evicted in set(self._estop_replay_cache.keys()) - set(ts_view.keys()):
                self._estop_replay_cache.pop(evicted, None)

            # F9-A per-issuer fairness check derived from cache contents.
            # No separate dict that drifts -- count entries owned by
            # ``issuer_id`` directly. After eviction this is naturally
            # correct: an attacker who flooded their cap and waited for
            # eviction now has fewer entries (eviction dropped them) and
            # can reclaim slots, which is the intended dynamic-attacker
            # rate-limit. A sustained attacker who paces floods to land
            # just after each eviction is bounded by ``per_issuer_cap``
            # at every instant -- they never hold more than that fraction
            # of the global cache, so legitimate operators always have
            # ``_resume_replay_cache_max() - per_issuer_cap`` slots available.
            issuer_slots = sum(1 for issuer, _mono in self._estop_replay_cache.values() if issuer == issuer_id)
            if issuer_slots >= per_issuer_cap:
                logger.warning(
                    "[safety] %s: REFUSED estop cache slot -- issuer %r already at cap %d "
                    "(F9-A per-issuer fairness bound; flood suspected)",
                    self.peer_id,
                    issuer_id,
                    per_issuer_cap,
                )
                # Audit the over-cap rejection so an operator dashboard
                # can alert on this. The replay-cache slot is NOT added,
                # but the lockout below still engages -- a legitimate
                # safety event is preserved even if the cache itself
                # cannot hold it.
                try:
                    self.publish_safety_event(
                        event_type="estop_per_issuer_cap_exceeded",
                        severity="warning",
                        payload={
                            "issuer": issuer_id,
                            "issuer_t": envelope_t,
                            "cap": per_issuer_cap,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug(
                        "[mesh] %s: estop_per_issuer_cap_exceeded audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
            else:
                self._estop_replay_cache[cache_key] = (issuer_id, now_mono)

            # F14-B (PR #195 review): when the issuer is over their
            # cap, refuse to engage the lockout AS WELL as refuse the
            # cache slot. Without this, a sustained attacker at-cap
            # could still drive the lockout to engage on every novel
            # `t` they emit -- defeating the F9-A fairness bound's
            # whole point (limiting one attacker's wall-clock impact).
            # An at-cap issuer's estops are dropped at the cache layer
            # AND the lockout layer; they remain audited as
            # `estop_per_issuer_cap_exceeded`.
            if issuer_slots >= per_issuer_cap:
                return

        if not self._estop_lockout.is_set():
            self._estop_lockout.set()
            self._last_estop_ts = time.time()
            sender = data.get("peer_id", "<remote>")
            logger.critical(
                "[safety] %s: lockout engaged via remote estop from %s",
                self.peer_id,
                sender,
            )
            self.publish_safety_event(
                event_type="remote_estop_engaged",
                severity="critical",
                payload={
                    "trigger": "remote",
                    "issuer": sender,
                    "issuer_t": data.get("t"),
                },
            )
        else:
            # F3: a second legitimate estop (different issuer, fresh ``t``)
            # arriving while the lockout is already engaged would otherwise
            # be silently dropped from the audit trail -- forensics lose
            # the signal that another operator also tried to engage.
            # Mirror the corroboration audit shape so every issuer of an
            # estop is preserved on the forensic record.
            try:
                self.publish_safety_event(
                    event_type="remote_estop_redundant",
                    severity="info",
                    payload={
                        "issuer": data.get("peer_id"),
                        "issuer_t": envelope_t,
                        "lockout_engaged_since": self._last_estop_ts,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug(
                    "[mesh] %s: remote_estop_redundant audit publish failed: %s",
                    self.peer_id,
                    audit_exc,
                )

    def _on_safety_resume(self, sample: Any) -> None:
        """Clear the local lockout in response to ``strands/safety/resume``.

        Wire authentication (mTLS + ACL) admits this handler. **When the
        operator supplies an ``STRANDS_MESH_ACL_FILE`` with role
        separation only ``operator_peer`` peers can publish here**; the
        default permissive ACL admits any cert-holding peer. Resume is
        further gated by the operator override code: the issuer signed
        ``HMAC(STRANDS_MESH_OVERRIDE_CODE, proof_nonce)`` and we
        recompute it locally; a mismatch means the issuer's override
        code differs from ours and we refuse. This is what stops one
        operator from clearing another operator's e-stop without
        explicit shared authorisation.

        Receivers without ``STRANDS_MESH_OVERRIDE_CODE`` configured
        FAIL CLOSED -- operators must distribute the code to every peer
        for fleet-wide remote resume to work.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            # Match _on_safety_estop's narrow exception tuple.
            # AttributeError: sample.payload is None or not bytes-like
            # UnicodeDecodeError: payload not valid UTF-8
            # json.JSONDecodeError: payload is not valid JSON
            # Wider exceptions (e.g. RuntimeError) bubble up and surface in logs
            # rather than silently locking the fleet into a half-state.
            return
        if not isinstance(data, dict):
            return

        local_code = os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip()
        if not local_code:
            logger.warning(
                "[safety] %s: refusing remote resume -- STRANDS_MESH_OVERRIDE_CODE "
                "not configured locally (operator code missing)",
                self.peer_id,
            )
            return

        proof_nonce = data.get("proof_nonce")
        provided_proof = data.get("override_proof")
        if not isinstance(proof_nonce, str) or not isinstance(provided_proof, str):
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing override_proof / proof_nonce",
                self.peer_id,
            )
            return

        expected_proof = hmac.new(
            local_code.encode(),
            proof_nonce.encode(),
            "sha256",
        ).hexdigest()
        if not hmac.compare_digest(expected_proof, provided_proof):
            logger.warning(
                "[safety] %s: refusing remote resume -- override_proof mismatch "
                "(issuer's STRANDS_MESH_OVERRIDE_CODE differs from local; constant-time compared)",
                self.peer_id,
            )
            return

        # Review feedback: freshness + replay cache.
        # The HMAC by itself authenticates the override code but says
        # nothing about when the envelope was minted -- a replay of a
        # captured envelope would still verify. Two cheap defences:
        #
        # 1. Freshness: reject envelopes whose ``t`` field is older
        #    than _resume_freshness_window_s() or more than the forward
        #    skew in the future. This matches the operator NTP
        #    requirement documented in CHANGELOG.
        # 2. Per-receiver replay cache: refuse a (issuer, proof_nonce)
        #    tuple we have already accepted within the freshness
        #    window. Bounded at _resume_replay_cache_max() entries.
        envelope_t = data.get("t")
        now = time.time()
        if not isinstance(envelope_t, (int, float)):
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing/invalid ``t``",
                self.peer_id,
            )
            return
        if envelope_t > now + _resume_forward_skew_s():
            logger.warning(
                "[safety] %s: refusing remote resume -- ``t``=%s in future (forward_skew_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                _resume_forward_skew_s(),
                now,
            )
            return
        if (now - envelope_t) > _resume_freshness_window_s():
            logger.warning(
                "[safety] %s: refusing remote resume -- ``t``=%s too old (freshness_window_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                _resume_freshness_window_s(),
                now,
            )
            return

        # Mirror the R20 estop strict-reject: an envelope without a
        # valid issuer peer_id would coalesce every "<unknown>"-issued
        # resume into a shared cache slot, polluting the bounded
        # ``_resume_replay_cache_max()`` allowance and giving any peer
        # who omits ``peer_id`` a free way to evict legitimate entries.
        # The canonical ``Mesh.emergency_stop`` issuer always sets
        # ``peer_id``; a malformed envelope is either a programming
        # bug or an attacker probing the cache.
        issuer_id = data.get("peer_id")
        if not isinstance(issuer_id, str) or not issuer_id:
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing/invalid ``peer_id``",
                self.peer_id,
            )
            return
        cache_key = (issuer_id, proof_nonce)
        with self._resume_replay_lock:
            if cache_key in self._resume_replay_cache:
                logger.warning(
                    "[safety] %s: REJECTED remote resume -- replay of (issuer=%s, proof_nonce=%s) already accepted",
                    self.peer_id,
                    issuer_id,
                    proof_nonce[:16] + "...",
                )
                # Audit the replay attempt -- this is exactly the
                # forensic signal an operator wants on a compromised
                # peer trying captured-and-replayed envelopes.
                try:
                    self.publish_safety_event(
                        event_type="resume_replay_rejected",
                        severity="warning",
                        payload={
                            "issuer": issuer_id,
                            "proof_nonce_prefix": proof_nonce[:16],
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # Audit publish is best-effort and must never block the
                    # safety path; narrow the catch tuple per AGENTS.md.
                    logger.debug(
                        "[mesh] %s: resume_replay_rejected audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
                return
            # R19: TTL math uses time.monotonic() (see PR #195 B5) --
            # envelope freshness above stays on time.time() because it
            # compares against the issuer wall clock; cache eviction is
            # local-only bookkeeping.
            now_mono = time.monotonic()
            _evict_replay_cache(
                self._resume_replay_cache,
                max_size=_resume_replay_cache_max(),
                ttl_s=_resume_freshness_window_s(),
                now_mono=now_mono,
            )
            self._resume_replay_cache[cache_key] = now_mono

        if self._estop_lockout.is_set():
            self._estop_lockout.clear()
            sender = data.get("peer_id", "<remote>")
            logger.warning("[safety] %s: lockout cleared via remote resume from %s", self.peer_id, sender)
            # R8-1: audit the receiver-side resume transition. Mirrors
            # _on_safety_estop above so verify_audit_integrity walkers
            # see the close of the lockout window for every peer that
            # entered one.
            self.publish_safety_event(
                event_type="remote_resume_applied",
                severity="info",
                payload={
                    "trigger": "remote",
                    "issuer": sender,
                    "issuer_t": data.get("t"),
                },
            )

    # RPC -- outgoing
    def send(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send a command to a single peer and return the first response.

        Phase-4 / D1 hardening: turn_id is a full 128-bit uuid4 (no
        truncation), and the expected responder is recorded so
        :meth:`_on_response` rejects forged responses from any peer
        other than *target*.

        R4-5: explicit guard against passing the
        :data:`BROADCAST_RESPONDER` sentinel (or any string containing a
        NUL byte) as ``target``. ``init_mesh``'s peer_id regex already
        rejects NUL on the receive side, so a real peer can't collide,
        but a future refactor that loosens that rule must not reopen
        the response-hijack surface that this method's contract closes.
        """
        if not self._running:
            return {"status": "error", "error": "mesh not running"}
        if not isinstance(target, str) or not target:
            return {"status": "error", "error": "send: target must be a non-empty string"}
        if "\x00" in target or target == BROADCAST_RESPONDER:
            return {
                "status": "error",
                "error": "send: target may not contain NUL or equal the BROADCAST_RESPONDER sentinel",
            }
        # R8-6: client-side validate before publishing. Prior to this fix,
        # programmatic callers (tests, third-party integrations, anything
        # that imports Mesh directly) skipped validate_command -- only the
        # robot_mesh tool path validated client-side. Receiver-side
        # _exec_cmd still validates, so this is defence-in-depth, but the
        # PR description and README claimed client-side AND server-side
        # validation; this closes the gap.
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: send to %s rejected client-side: %s", self.peer_id, target, exc)
            return {"status": "error", "error": f"validation: {exc}"}
        # 128-bit turn id -- at 32 bits the birthday-collision window
        # under heavy concurrent RPC load was practical (~65k turns
        # before 50% collision); 128 bits removes that surface entirely.
        turn = uuid.uuid4().hex
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
            # R5-defensive: belt-and-suspenders. The public guard above
            # already rejects target == BROADCAST_RESPONDER and target
            # containing NUL, but a future refactor that adds another
            # path into this method (e.g. an internal helper that bypasses
            # the public guard) must not reopen the response-hijack
            # surface. Re-checking here makes the invariant explicit at
            # the assignment site.
            if target == BROADCAST_RESPONDER or "\x00" in target:
                self._pending.pop(turn, None)
                self._responses.pop(turn, None)
                raise ValueError("send: target may not equal BROADCAST_RESPONDER or contain NUL")
            self._expected_responders[turn] = target
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self.publish(f"strands/{target}/cmd", msg)
            event.wait(timeout=timeout)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
                self._expected_responders.pop(turn, None)
        return resps[0] if resps else {"status": "timeout"}

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        """Broadcast a command to every peer and return all responses.

        Phase-4 / D1: turn_id is a full 128-bit uuid4 (no truncation).
        Broadcast turns accept responses from any responder by design,
        so the responder_id check is bypassed (sentinel
        ``BROADCAST_RESPONDER``).
        """
        if not self._running:
            return []
        # R8-6: client-side validate before publishing. broadcast()'s
        # return type is list[dict] (responses), so a validation failure
        # has no structured slot -- log the rejection and return [] so
        # callers see "no responses" rather than a partial broadcast.
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: broadcast rejected client-side: %s", self.peer_id, exc)
            return []
        turn = uuid.uuid4().hex
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
            # Sentinel -- broadcast accepts responses from any peer.
            self._expected_responders[turn] = BROADCAST_RESPONDER
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self.publish("strands/broadcast", msg)
            event.wait(timeout=timeout)
            time.sleep(0.3)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
                self._expected_responders.pop(turn, None)
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

        self.publish(
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
        timeout -- useful for telemetry (which peers acknowledged before the
        stop fanned out).
        """
        self._estop_lockout.set()
        self._last_estop_ts = time.time()
        responses = self.broadcast({"action": "stop"}, timeout=3.0)
        self.publish(
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
        logger.critical("[safety] %s: EMERGENCY STOP engaged -- lockout active", self.peer_id)
        return responses

    def _resume_lockout(self, override_code: str) -> dict[str, Any]:
        """Clear the emergency-stop lockout if *override_code* matches.

        Compared in constant time against ``STRANDS_MESH_OVERRIDE_CODE``.

        R5-4: the wire response is a single generic shape (``{"status":
        "ok"}`` on success, ``{"status": "error", "error": "resume
        rejected"}`` on every failure including "lockout not engaged" and
        "override code unconfigured") so a remote prober cannot use
        differential responses as oracles for:

        * whether the lockout is engaged at all (``noop`` vs ``error``),
        * whether ``STRANDS_MESH_OVERRIDE_CODE`` is configured (``not
          configured`` vs ``invalid code``),
        * how long the lockout was held (``lockout_elapsed_s``).

        Structured detail is preserved in the local
        ``publish_safety_event`` audit record where forensics can use it.
        Local callers (e.g. operator tooling that wants to show "already
        unlocked" UI) can still distinguish via the local audit log.
            {"status": "error", "error": "<reason>"}               # rejected

        Every attempt -- successful or not -- is recorded in the audit log
        through :meth:`publish_safety_event`.
        """
        # R5-4: every non-success path returns the same generic dict so
        # a remote caller cannot use the response shape as an oracle.
        # Structured rejection reasons are preserved in the local audit
        # log via publish_safety_event.
        _generic_error = {"status": "error", "error": "resume rejected"}

        if not self._estop_lockout.is_set():
            self.publish_safety_event(
                event_type="resume_denied",
                severity="info",
                payload={"sender_id": self.peer_id, "reason": "lockout not engaged"},
            )
            return _generic_error

        expected = os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip()
        provided = (override_code or "").strip()
        if not expected:
            self.publish_safety_event(
                event_type="resume_denied",
                severity="warning",
                payload={"sender_id": self.peer_id, "reason": "STRANDS_MESH_OVERRIDE_CODE not configured"},
            )
            return _generic_error
        # Constant-time compare so an attacker probing the override code
        # cannot use response-time variation to learn it byte-by-byte.
        if not hmac.compare_digest(expected.encode(), provided.encode()):
            self.publish_safety_event(
                event_type="resume_denied",
                severity="warning",
                payload={"sender_id": self.peer_id, "reason": "bad override code"},
            )
            return _generic_error

        elapsed = time.time() - self._last_estop_ts
        self._estop_lockout.clear()
        self.publish_safety_event(
            event_type="resume_ok",
            severity="info",
            payload={"sender_id": self.peer_id, "lockout_elapsed_s": elapsed},
        )

        # R8-5: bind a proof-of-override-code into the resume envelope so
        # receivers can re-verify on _on_safety_resume. Without this,
        # any operator-class peer could fan-out a resume just by virtue
        # of being on the ACL; the override code adds a second factor
        # that every receiver re-verifies by recomputing
        # HMAC(local_code, proof_nonce).
        #
        # The proof_nonce is per-resume (uuid4.hex). We deliberately do
        # NOT include the override code itself in the published payload
        # or the audit log -- only the HMAC of (code, nonce).
        proof_nonce = uuid.uuid4().hex
        override_proof = hmac.new(
            expected.encode(),
            proof_nonce.encode(),
            "sha256",
        ).hexdigest()
        self.publish(
            "strands/safety/resume",
            {
                "peer_id": self.peer_id,
                "t": time.time(),
                "lockout_elapsed_s": elapsed,
                "proof_nonce": proof_nonce,
                "override_proof": override_proof,
            },
        )
        logger.warning("[safety] %s: resume after %.1fs lockout", self.peer_id, elapsed)
        # R5-4: success is also generic on the wire; the local audit
        # record (resume_ok above) carries the elapsed time for forensics.
        return {"status": "ok"}

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        """Publish *payload* on *key* via the mesh transport.

        Wire authentication is owned by the Zenoh transport: outbound
        bytes ride a TLS link whose cert binds the peer identity, and
        the ACL gates which key-expressions this peer can publish on.
        This method simply forwards to ``put()`` -- it stays as a
        single chokepoint so a future hook (audit, telemetry,
        compression) can land in one place.

        Renamed from ``_put_signed`` after the application-layer signing
        envelope was dropped (commit 7113742). The old name was a
        historical artefact: nothing in the body ever signed anything
        once Zenoh's mTLS + ACL took over identity and authorization.
        """
        put(key, payload)


# init_mesh -- the only public constructor
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
