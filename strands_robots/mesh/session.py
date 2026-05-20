"""Shared Zenoh session and peer registry for the mesh networking layer.

This module provides a single, ref-counted :func:`zenoh.open` session per process
and a thread-safe registry of discovered peers.  It is the lowest layer of the
mesh stack — higher-level constructs (``Mesh``, presence, RPC) build on top.

The Zenoh dependency is **lazy**: ``import strands_robots.mesh_session`` does not
import ``zenoh`` at module level.  The first call to :func:`get_session` triggers
the real import.  If ``eclipse-zenoh`` is not installed the function returns
``None`` and all publish helpers become safe no-ops.

Connection strategy (when no explicit endpoint is configured):

1. Try to **listen** on ``tcp/127.0.0.1:{STRANDS_MESH_PORT}`` — this makes the
   first process the local router.
2. If the port is already bound, fall back to **client** mode and connect to the
   same endpoint.
3. Zenoh scouting (multicast) handles LAN discovery automatically.

Environment variables
---------------------
``ZENOH_CONNECT``
    Comma-separated remote endpoint(s) — e.g. ``tcp/10.0.0.1:7447``.
``ZENOH_LISTEN``
    Comma-separated listen endpoint(s).
``STRANDS_MESH_PORT``
    Local auto-mesh port (default ``7447``).
``STRANDS_MESH``
    Set to ``false`` to disable mesh globally.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Session singleton — one ``zenoh.Session`` per process, ref-counted


_SESSION: Any | None = None  # zenoh.Session when open, else None
_SESSION_LOCK = threading.Lock()
_SESSION_REFS: int = 0


# Constants


#: Default heartbeat frequency (Hz).  Presence payloads are published at this rate.
HEARTBEAT_HZ: float = 2.0

#: Default state-publishing frequency (Hz).
STATE_HZ: float = 10.0

#: Default camera-publishing frequency (Hz).  ``0`` disables the camera
#: loop — opt-in via the ``STRANDS_MESH_CAMERA_HZ`` environment variable
#: because frames are large and bandwidth-heavy.
CAMERA_HZ: float = 0.0

#: Seconds without a heartbeat before a peer is considered dead.
PEER_TIMEOUT: float = 10.0
#: Pose publishing frequency (Hz).  Publishes SE(3) pose when a pose
#: provider (SLAM, odometry, VIO) is available on the robot.
POSE_HZ: float = 10.0

#: IMU publishing frequency (Hz).  Downsampled from hardware rate.
IMU_HZ: float = 10.0

#: Odometry publishing frequency (Hz).
ODOM_HZ: float = 10.0

#: Health/fleet-monitoring publishing frequency (Hz).
HEALTH_HZ: float = 0.5

#: LiDAR summary publishing frequency (Hz).
LIDAR_SUMMARY_HZ: float = 5.0

#: LiDAR state publishing frequency (Hz).
LIDAR_STATE_HZ: float = 1.0

#: Hand/end-effector state publishing frequency (Hz).
HAND_HZ: float = 50.0

#: Map info publishing frequency (Hz).
MAP_INFO_HZ: float = 0.2


# Backend selection helpers — when STRANDS_MESH_BACKEND is "iot" or "bridge",
# get_session() / put() / current_session() / session_alive() delegate to the
# transport factory instead of opening a Zenoh session directly. The "zenoh"
# default keeps the historical behaviour byte-for-byte so the 200+ existing
# mesh tests pass unmodified.


def _backend_choice() -> str:
    """Read STRANDS_MESH_BACKEND. Defaults to ``zenoh``. Unknown values fall
    back to ``zenoh`` (matches strands_robots.mesh.transport.factory)."""
    raw = os.getenv("STRANDS_MESH_BACKEND", "zenoh").strip().lower()
    if raw not in ("zenoh", "iot", "bridge"):
        return "zenoh"
    return raw


def _is_transport_backend() -> bool:
    """True when the backend is anything other than the legacy zenoh path."""
    return _backend_choice() in ("iot", "bridge")


# PeerInfo


@dataclass
class PeerInfo:
    """A discovered peer on the Zenoh mesh.

    Attributes:
        peer_id: Unique identifier for this peer (e.g. ``"so100-a1b2"``).
        peer_type: One of ``"robot"``, ``"sim"``, or ``"agent"``.
        hostname: The hostname the peer reported.
        last_seen: :func:`time.time` of the most recent heartbeat.
        caps: Arbitrary capability dictionary broadcast in the presence payload.
    """

    peer_id: str
    peer_type: str = "robot"
    hostname: str = ""
    last_seen: float = 0.0
    caps: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds since the last heartbeat."""
        return time.time() - self.last_seen

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "peer_id": self.peer_id,
            "type": self.peer_type,
            "hostname": self.hostname,
            "age": round(self.age, 1),
            **self.caps,
        }

    def __repr__(self) -> str:
        return f"PeerInfo(peer_id={self.peer_id!r}, type={self.peer_type!r}, age={self.age:.1f}s)"


# Peer registry — shared across all Mesh instances in the same process


_PEERS: dict[str, PeerInfo] = {}
_PEERS_VERSION: int = 0
_PEERS_LOCK = threading.Lock()


def update_peer(peer_id: str, peer_type: str, hostname: str, caps: dict[str, Any]) -> bool:
    """Insert or update a peer.  Returns ``True`` when the peer is new."""
    global _PEERS_VERSION  # noqa: PLW0603 — module-level singleton by design
    with _PEERS_LOCK:
        is_new = peer_id not in _PEERS
        _PEERS[peer_id] = PeerInfo(
            peer_id=peer_id,
            peer_type=peer_type,
            hostname=hostname,
            last_seen=time.time(),
            caps=caps,
        )
        if is_new:
            _PEERS_VERSION += 1
        return is_new


def prune_peers(timeout: float = PEER_TIMEOUT) -> list[str]:
    """Remove peers that have not sent a heartbeat within *timeout* seconds.

    Returns:
        List of pruned peer IDs (may be empty).
    """
    global _PEERS_VERSION  # noqa: PLW0603
    now = time.time()
    pruned: list[str] = []
    with _PEERS_LOCK:
        stale = [pid for pid, p in _PEERS.items() if now - p.last_seen > timeout]
        for pid in stale:
            del _PEERS[pid]
            _PEERS_VERSION += 1
            pruned.append(pid)
    for pid in pruned:
        logger.info("Mesh: peer %s timed out", pid)
    return pruned


def get_peers() -> list[dict[str, Any]]:
    """Return all known peers as plain dicts."""
    with _PEERS_LOCK:
        return [p.to_dict() for p in _PEERS.values()]


def get_peer(peer_id: str) -> dict[str, Any] | None:
    """Return a single peer by *peer_id*, or ``None`` if unknown."""
    with _PEERS_LOCK:
        p = _PEERS.get(peer_id)
        return p.to_dict() if p else None


def peer_count() -> int:
    """Number of currently known (non-stale) peers."""
    with _PEERS_LOCK:
        return len(_PEERS)


def clear_peers() -> None:
    """Remove **all** peers.  Intended for tests only."""
    global _PEERS_VERSION  # noqa: PLW0603
    with _PEERS_LOCK:
        _PEERS.clear()
        _PEERS_VERSION += 1


# Session lifecycle


def _build_config() -> Any:
    """Create a ``zenoh.Config`` from environment variables.

    Returns:
        A ``zenoh.Config`` instance.

    Raises:
        ImportError: If ``eclipse-zenoh`` is not installed.
    """
    import zenoh

    config = zenoh.Config()

    connect = os.getenv("ZENOH_CONNECT")
    listen = os.getenv("ZENOH_LISTEN")

    if connect:
        endpoints = [e.strip() for e in connect.split(",")]
        config.insert_json5("connect/endpoints", json.dumps(endpoints))
    if listen:
        endpoints = [e.strip() for e in listen.split(",")]
        config.insert_json5("listen/endpoints", json.dumps(endpoints))

    return config


def current_session() -> Any | None:
    """Return the existing session/transport without bumping the refcount.

    Backend-aware: returns the active transport singleton when
    ``STRANDS_MESH_BACKEND`` is ``iot`` / ``bridge``, otherwise the raw
    Zenoh session (legacy behaviour).
    """
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        return current_transport()

    with _SESSION_LOCK:
        return _SESSION


def get_session() -> Any | None:
    """Acquire the shared mesh transport (lazy, ref-counted).

    Backend selection comes from ``STRANDS_MESH_BACKEND``:

    - ``zenoh`` (default) — open / reuse a ``zenoh.Session`` exactly as before.
      Returned object is the raw session; callers can ``.declare_subscriber()``
      on it.
    - ``iot`` / ``bridge`` — delegate to
      :mod:`strands_robots.mesh.transport.factory`; the returned object is an
      :class:`~strands_robots.mesh.transport.IotMqttTransport` or
      :class:`~strands_robots.mesh.transport.BridgeTransport` which **also**
      exposes ``put()`` / ``declare_subscriber()`` / ``close()`` so existing
      Mesh code works unchanged.

    Returns:
        Backend-dependent: ``zenoh.Session``, ``IotMqttTransport``,
        ``BridgeTransport``, or ``None`` if the chosen backend is unavailable.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    if _is_transport_backend():
        # Delegate to the transport factory. The factory holds its own
        # refcount independently of _SESSION_REFS — that's fine, callers
        # that release_session() will see the matching release_transport().
        from strands_robots.mesh.transport.factory import get_transport

        return get_transport()

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh  # noqa: F811 — lazy import
        except ImportError:
            logger.debug("eclipse-zenoh not installed — mesh disabled")
            return None

        # STRANDS_MESH_PORT is read at session-open time so a process can be
        # configured via env vars without re-importing.  Bad input falls back
        # to the default and warns once — never raises (the default behaviour
        # is to keep the mesh quietly off rather than crash the host robot).
        port_env = os.getenv("STRANDS_MESH_PORT", "7447")
        try:
            mesh_port = int(port_env)
            if not (1 <= mesh_port <= 65535):
                raise ValueError(f"port {mesh_port} out of range")
        except ValueError as exc:
            logger.warning(
                "Invalid STRANDS_MESH_PORT=%r (%s) — falling back to 7447",
                port_env,
                exc,
            )
            mesh_port = 7447
        local_ep = f"tcp/127.0.0.1:{mesh_port}"

        connect_env = os.getenv("ZENOH_CONNECT")
        listen_env = os.getenv("ZENOH_LISTEN")

        # When no explicit endpoints are set, try to become the local router.
        if not connect_env and not listen_env:
            try:
                cfg = zenoh.Config()
                cfg.insert_json5("listen/endpoints", json.dumps([local_ep]))
                cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except Exception as exc:  # noqa: BLE001 — fall back to client mode
                # Port already bound (the most common case) is not an error.
                # Log at debug so a real misconfiguration (e.g. bad iface) can
                # still be diagnosed without spamming WARNING during the
                # normal "second process joining the mesh" flow.
                logger.debug(
                    "Zenoh listener on %s unavailable (%s) — trying client mode",
                    local_ep,
                    exc,
                )

            # Fall back to client mode — connect to the existing listener.
            try:
                cfg = _build_config()
                cfg.insert_json5("mode", '"client"')
                cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (client → %s)", local_ep)
                return _SESSION
            except Exception as exc:
                logger.warning("Zenoh session open failed (client mode): %s", exc)
                return None

        # Explicit endpoints provided via env vars.
        try:
            cfg = _build_config()
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except Exception as exc:
            logger.warning("Zenoh session open failed: %s", exc)
            return None


def _get_zenoh_session_directly() -> Any | None:
    """Open/reuse the Zenoh session directly, bypassing transport-backend routing.

    This is used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    when it is instantiated as part of a :class:`BridgeTransport`. In that scenario,
    ``get_session()`` would re-enter the factory's ``_LOCK`` (since
    ``_is_transport_backend()`` returns True for bridge mode) causing a deadlock.

    This function always goes through the raw Zenoh path regardless of
    ``STRANDS_MESH_BACKEND``. It shares the same ``_SESSION`` singleton and
    ``_SESSION_LOCK``.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh
        except ImportError:
            logger.debug("eclipse-zenoh not installed — mesh disabled")
            return None

        port_env = os.getenv("STRANDS_MESH_PORT", "7447")
        try:
            mesh_port = int(port_env)
            if not (1 <= mesh_port <= 65535):
                raise ValueError(f"port {mesh_port} out of range")
        except ValueError as exc:
            logger.warning(
                "Invalid STRANDS_MESH_PORT=%r (%s) — falling back to 7447",
                port_env,
                exc,
            )
            mesh_port = 7447
        local_ep = f"tcp/127.0.0.1:{mesh_port}"

        connect_env = os.getenv("ZENOH_CONNECT")
        listen_env = os.getenv("ZENOH_LISTEN")

        if not connect_env and not listen_env:
            try:
                cfg = zenoh.Config()
                cfg.insert_json5("listen/endpoints", json.dumps([local_ep]))
                cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Zenoh listener on %s unavailable (%s) — trying client mode",
                    local_ep,
                    exc,
                )

            try:
                cfg = _build_config()
                cfg.insert_json5("mode", '"client"')
                cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (client → %s)", local_ep)
                return _SESSION
            except Exception as exc:
                logger.warning("Zenoh session open failed (client mode): %s", exc)
                return None

        try:
            cfg = _build_config()
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except Exception as exc:
            logger.warning("Zenoh session open failed: %s", exc)
            return None


def release_session() -> None:
    """Release one reference to the shared mesh session.

    Delegates to the transport factory when the active backend is
    ``iot`` / ``bridge``; otherwise falls back to the legacy Zenoh refcount.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import release_transport

        release_transport()
        return

    with _SESSION_LOCK:
        if _SESSION_REFS <= 0:
            return
        _SESSION_REFS -= 1
        if _SESSION_REFS <= 0 and _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None
            _SESSION_REFS = 0
            logger.info("Zenoh mesh session closed")


def session_alive() -> bool:
    """Return ``True`` if the current backend's session/transport is open."""
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        t = current_transport()
        return t is not None and t.is_alive()

    with _SESSION_LOCK:
        return _SESSION is not None


# Publish helper


def put(key: str, data: dict[str, Any]) -> None:
    """Publish a JSON payload to the mesh.

    Fire-and-forget. No-op when no session/transport is open.

    Backend-aware: delegates to the active transport's ``put()`` when
    running under ``STRANDS_MESH_BACKEND=iot`` / ``bridge``; otherwise
    encodes JSON and pushes to the Zenoh session directly (legacy path).
    """
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        t = current_transport()
        if t is None:
            return
        try:
            t.put(key, data)
        except Exception as exc:
            logger.debug("Mesh transport put error on %s: %s", key, exc)
        return

    if _SESSION is None:
        return
    try:
        _SESSION.put(key, json.dumps(data).encode())
    except Exception as exc:
        logger.debug("Zenoh put error on %s: %s", key, exc)


# Process cleanup


def _atexit_cleanup() -> None:
    """Best-effort session teardown on process exit."""
    global _SESSION, _SESSION_REFS  # noqa: PLW0603
    with _SESSION_LOCK:
        if _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None
            _SESSION_REFS = 0


atexit.register(_atexit_cleanup)


def _session_alive_directly() -> bool:
    """Return ``True`` if the raw Zenoh session is open, bypassing backend routing.

    Used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    to avoid recursion when operating inside a :class:`BridgeTransport`.
    """
    with _SESSION_LOCK:
        return _SESSION is not None


def _current_zenoh_session_directly() -> Any | None:
    """Return the raw Zenoh session without bumping refcount, bypassing backend routing.

    Used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    to avoid recursion when operating inside a :class:`BridgeTransport`.
    """
    with _SESSION_LOCK:
        return _SESSION
