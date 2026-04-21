#!/usr/bin/env python3
"""
Zenoh session singleton — one session per process, ref-counted.

Every Mesh instance shares the same ``zenoh.Session`` to avoid
duplicating discovery traffic and file descriptors.  The session is
opened lazily on the first ``MeshSession.open()`` call and closed
when the last consumer calls ``MeshSession.close()`` (or the process
exits via the ``atexit`` hook).

Fork safety
-----------
If the process is forked (``os.fork``), child processes get a stale
session.  ``MeshSession.open()`` detects PID changes and re-initialises
the session automatically.

Connection config
-----------------
By default Zenoh uses **multicast scouting** for peer discovery on the
local LAN.  Override with environment variables::

    # Connect to a specific endpoint (WAN / CI / Docker)
    ZENOH_CONNECT=tcp/192.168.1.10:7447

    # Listen on an explicit endpoint
    ZENOH_LISTEN=tcp/0.0.0.0:7447

    # Disable mesh entirely
    STRANDS_MESH=false
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MeshSession:
    """Process-level singleton over ``zenoh.Session``.

    Thread-safe, ref-counted, fork-aware.

    Usage::

        session = MeshSession.open()   # refcount +1
        # ... use session ...
        MeshSession.close()            # refcount -1; actual close at 0
    """

    _lock = threading.Lock()
    _session: Any = None  # zenoh.Session (typed as Any to avoid import)
    _refcount: int = 0
    _pid: int | None = None
    _atexit_registered: bool = False

    @classmethod
    def open(cls, config_overrides: dict[str, Any] | None = None) -> Any:
        """Acquire the shared Zenoh session.

        Creates the session on first call.  Subsequent calls increment
        the reference count and return the same session.

        Args:
            config_overrides: Optional dict of Zenoh config JSON5 paths
                to values.  Example::

                    {"connect/endpoints": ["tcp/127.0.0.1:7447"]}

                Overrides are applied *after* environment-variable config.

        Returns:
            A ``zenoh.Session`` instance, or ``None`` if eclipse-zenoh
            is not installed.

        Raises:
            RuntimeError: If the Zenoh session cannot be opened after
                applying configuration.
        """
        with cls._lock:
            # Fork detection: if PID changed, the session is stale.
            current_pid = os.getpid()
            if cls._session is not None and cls._pid != current_pid:
                logger.warning(
                    "PID changed (%s → %s) — re-initialising Zenoh session (probable fork). Old session abandoned.",
                    cls._pid,
                    current_pid,
                )
                # Don't close the parent's session — just discard our ref.
                cls._session = None
                cls._refcount = 0

            if cls._session is not None:
                cls._refcount += 1
                return cls._session

            # Lazy import — avoid pulling zenoh at strands_robots import time.
            try:
                import zenoh
            except ImportError:
                logger.debug(
                    "eclipse-zenoh not installed — mesh disabled.  Install with: pip install strands-robots[mesh]"
                )
                return None

            cfg = zenoh.Config()

            # --- Environment-variable overrides ---
            connect = os.getenv("ZENOH_CONNECT")
            if connect:
                endpoints = [e.strip() for e in connect.split(",")]
                try:
                    cfg.insert_json5("connect/endpoints", json.dumps(endpoints))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to apply ZENOH_CONNECT=%s: %s", connect, exc)

            listen = os.getenv("ZENOH_LISTEN")
            if listen:
                endpoints = [e.strip() for e in listen.split(",")]
                try:
                    cfg.insert_json5("listen/endpoints", json.dumps(endpoints))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to apply ZENOH_LISTEN=%s: %s", listen, exc)

            # --- Programmatic overrides ---
            if config_overrides:
                for path, value in config_overrides.items():
                    try:
                        cfg.insert_json5(path, json.dumps(value))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to apply config override %s=%r: %s",
                            path,
                            value,
                            exc,
                        )

            try:
                cls._session = zenoh.open(cfg)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to open Zenoh session: {exc}.  Check ZENOH_CONNECT / ZENOH_LISTEN env vars."
                ) from exc

            cls._refcount = 1
            cls._pid = current_pid

            if not cls._atexit_registered:
                atexit.register(cls._atexit_cleanup)
                cls._atexit_registered = True

            logger.info("Zenoh mesh session opened (pid=%s)", current_pid)
            return cls._session

    @classmethod
    def close(cls) -> None:
        """Release one reference to the shared session.

        When the reference count reaches zero the underlying
        ``zenoh.Session`` is closed.
        """
        with cls._lock:
            if cls._refcount <= 0:
                return

            cls._refcount -= 1
            if cls._refcount == 0 and cls._session is not None:
                try:
                    cls._session.close()
                except Exception:  # noqa: BLE001
                    pass  # Best-effort; session may already be dead.
                cls._session = None
                cls._pid = None
                logger.info("Zenoh mesh session closed (refcount → 0)")

    @classmethod
    def _atexit_cleanup(cls) -> None:
        """Best-effort cleanup at interpreter shutdown."""
        with cls._lock:
            if cls._session is not None:
                try:
                    cls._session.close()
                except Exception:  # noqa: BLE001
                    pass
                cls._session = None
                cls._refcount = 0
                cls._pid = None

    # --- Introspection helpers (testing / debugging) ---

    @classmethod
    def is_open(cls) -> bool:
        """Return ``True`` if a session is currently open."""
        with cls._lock:
            return cls._session is not None

    @classmethod
    def refcount(cls) -> int:
        """Return the current reference count."""
        with cls._lock:
            return cls._refcount

    @classmethod
    def _reset(cls) -> None:
        """Force-reset internal state.  **Testing only.**"""
        with cls._lock:
            if cls._session is not None:
                try:
                    cls._session.close()
                except Exception:  # noqa: BLE001
                    pass
            cls._session = None
            cls._refcount = 0
            cls._pid = None
