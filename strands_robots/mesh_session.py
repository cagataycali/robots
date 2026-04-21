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
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session singleton — one ``zenoh.Session`` per process, ref-counted
# ---------------------------------------------------------------------------

_SESSION: Any | None = None  # zenoh.Session when open, else None
_SESSION_LOCK = threading.Lock()
_SESSION_REFS: int = 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default heartbeat frequency (Hz).  Presence payloads are published at this rate.
HEARTBEAT_HZ: float = 2.0

#: Default state-publishing frequency (Hz).
STATE_HZ: float = 10.0

#: Seconds without a heartbeat before a peer is considered dead.
PEER_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# PeerInfo
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Peer registry — shared across all Mesh instances in the same process
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def _build_config() -> Any:
    """Create a ``zenoh.Config`` from environment variables.

    Returns:
        A ``zenoh.Config`` instance.

    Raises:
        ImportError: If ``eclipse-zenoh`` is not installed.
    """
    import os

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


def get_session() -> Any | None:
    """Acquire the shared Zenoh session (lazy, ref-counted).

    * First call opens the session; subsequent calls increment the refcount.
    * If ``eclipse-zenoh`` is not installed, returns ``None``.
    * Thread-safe.

    Returns:
        An open ``zenoh.Session``, or ``None`` if Zenoh is unavailable.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh  # noqa: F811 — lazy import
        except ImportError:
            logger.debug("eclipse-zenoh not installed — mesh disabled")
            return None

        import os

        mesh_port = int(os.getenv("STRANDS_MESH_PORT", "7447"))
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
            except Exception:
                # Port already bound — another process is the local router.
                pass

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


def release_session() -> None:
    """Release one reference to the shared session.

    When the refcount reaches zero the underlying ``zenoh.Session`` is closed.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

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
    """Return ``True`` if a Zenoh session is currently open."""
    with _SESSION_LOCK:
        return _SESSION is not None


# ---------------------------------------------------------------------------
# Publish helper
# ---------------------------------------------------------------------------


def put(key: str, data: dict[str, Any]) -> None:
    """Publish a JSON payload to the mesh.

    This is a fire-and-forget helper.  If no session is open the call is a
    no-op (no exception raised).

    Args:
        key: Zenoh key expression (e.g. ``"strands/picker/presence"``).
        data: JSON-serialisable dictionary.
    """
    if _SESSION is None:
        return
    try:
        _SESSION.put(key, json.dumps(data).encode())
    except Exception as exc:
        logger.debug("Zenoh put error on %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Process cleanup
# ---------------------------------------------------------------------------


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
