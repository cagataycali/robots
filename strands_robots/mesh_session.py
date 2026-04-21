"""Zenoh session singleton — ONE session per process, ref-counted.

This module manages a shared ``zenoh.Session`` so that multiple ``Mesh``
instances (one per Robot/Simulation) reuse a single network socket.

Session lifecycle::

    session = get_session()         # ref +1, opens on first call
    session2 = get_session()        # ref +1, returns same session
    release_session()               # ref -1
    release_session()               # ref → 0, session.close()

Environment variables
---------------------
ZENOH_CONNECT
    Comma-separated Zenoh endpoints to connect to.
    Example: ``tcp/10.0.0.5:7447,tcp/10.0.0.6:7447``
ZENOH_LISTEN
    Comma-separated Zenoh endpoints to listen on.
    Example: ``tcp/0.0.0.0:7447``
STRANDS_MESH_PORT
    Local auto-mesh port (default 7447).  The first process on a host
    listens; subsequent processes connect as clients.
STRANDS_MESH
    Set to ``false`` to disable mesh globally.

Requires ``pip install strands-robots[mesh]`` (eclipse-zenoh).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MESH_PORT = 7447


@dataclass(frozen=True)
class MeshConfig:
    """Immutable Zenoh connection configuration.

    Attributes:
        connect: Zenoh endpoints to connect to (e.g. ``("tcp/10.0.0.5:7447",)``).
        listen: Zenoh endpoints to listen on (e.g. ``("tcp/0.0.0.0:7447",)``).
        port: Local auto-mesh port used when neither *connect* nor *listen*
              is specified.  Default ``7447``.
    """

    connect: tuple[str, ...] = ()
    listen: tuple[str, ...] = ()
    port: int = DEFAULT_MESH_PORT

    @classmethod
    def from_env(cls) -> MeshConfig:
        """Build a ``MeshConfig`` from environment variables.

        Reads ``ZENOH_CONNECT``, ``ZENOH_LISTEN``, and
        ``STRANDS_MESH_PORT``.  Missing variables produce empty tuples /
        the default port.
        """
        connect_raw = os.getenv("ZENOH_CONNECT", "")
        listen_raw = os.getenv("ZENOH_LISTEN", "")
        port = int(os.getenv("STRANDS_MESH_PORT", str(DEFAULT_MESH_PORT)))

        connect = tuple(e.strip() for e in connect_raw.split(",") if e.strip()) if connect_raw else ()
        listen = tuple(e.strip() for e in listen_raw.split(",") if e.strip()) if listen_raw else ()

        return cls(connect=connect, listen=listen, port=port)


# ---------------------------------------------------------------------------
# Session singleton
# ---------------------------------------------------------------------------

_SESSION: Any = None  # Optional[zenoh.Session] — typed as Any to avoid import-time dep
_SESSION_LOCK = threading.Lock()
_SESSION_REFS: int = 0


def _apply_config(zenoh_config: Any, config: MeshConfig) -> None:
    """Apply *config* endpoints to a ``zenoh.Config`` object.

    Mutates *zenoh_config* in-place via ``insert_json5``.
    """
    if config.connect:
        zenoh_config.insert_json5("connect/endpoints", json.dumps(list(config.connect)))
    if config.listen:
        zenoh_config.insert_json5("listen/endpoints", json.dumps(list(config.listen)))


def get_session(config: MeshConfig | None = None) -> Any:
    """Acquire the shared Zenoh session (lazy, ref-counted).

    On the first call the session is opened.  Subsequent calls increment
    the reference count and return the same session.

    When neither ``ZENOH_CONNECT`` nor ``ZENOH_LISTEN`` are set, *auto-mesh*
    kicks in: the first process on the host listens on
    ``tcp/127.0.0.1:{port}``; later processes connect as clients.

    Parameters
    ----------
    config:
        Optional explicit configuration.  If ``None``, reads from
        environment variables via :meth:`MeshConfig.from_env`.

    Returns
    -------
    zenoh.Session | None
        The shared session, or ``None`` if eclipse-zenoh is not installed
        or the global kill-switch ``STRANDS_MESH=false`` is set.
    """
    global _SESSION, _SESSION_REFS

    # Global kill switch
    if os.getenv("STRANDS_MESH", "true").lower() == "false":
        return None

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        # Lazy import — zenoh is optional
        try:
            import importlib

            zenoh = importlib.import_module("zenoh")
        except ImportError:
            logger.debug("eclipse-zenoh not installed — mesh disabled (pip install strands-robots[mesh])")
            return None

        if config is None:
            config = MeshConfig.from_env()

        # If explicit endpoints are configured, use them directly
        if config.connect or config.listen:
            zenoh_config = zenoh.Config()
            _apply_config(zenoh_config, config)
            _SESSION = zenoh.open(zenoh_config)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened (explicit config)")
            return _SESSION

        # Auto-mesh: try listen+connect on localhost (first process wins)
        mesh_ep = f"tcp/127.0.0.1:{config.port}"

        try:
            cfg_listen = zenoh.Config()
            cfg_listen.insert_json5("listen/endpoints", json.dumps([mesh_ep]))
            cfg_listen.insert_json5("connect/endpoints", json.dumps([mesh_ep]))
            _SESSION = zenoh.open(cfg_listen)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened (listener on %s)", mesh_ep)
            return _SESSION
        except Exception:
            # Port already taken — another process is listening
            pass

        # Connect as client to the existing listener
        cfg_client = zenoh.Config()
        cfg_client.insert_json5("mode", '"client"')
        cfg_client.insert_json5("connect/endpoints", json.dumps([mesh_ep]))
        _SESSION = zenoh.open(cfg_client)
        _SESSION_REFS = 1
        logger.info("Zenoh mesh session opened (client → %s)", mesh_ep)
        return _SESSION


def release_session() -> None:
    """Release one reference to the shared session.

    When the reference count reaches zero the session is closed.
    """
    global _SESSION, _SESSION_REFS

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


def session_info() -> dict[str, Any]:
    """Return diagnostic info about the current session state.

    Useful for dashboards and debugging.  Does not acquire or release
    the session.
    """
    with _SESSION_LOCK:
        return {
            "active": _SESSION is not None,
            "refs": _SESSION_REFS,
        }
