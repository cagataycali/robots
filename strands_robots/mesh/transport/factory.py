"""Process-wide :class:`MeshTransport` factory and ref-counted singleton.

Mirrors the existing :func:`~strands_robots.mesh.session.get_session` /
:func:`~strands_robots.mesh.session.release_session` pair so :class:`Mesh`
can swap transports by setting ``STRANDS_MESH_BACKEND`` without changing
its lifecycle code.

Backend selection
-----------------
Selection is done at the first :func:`get_transport` call:

- ``zenoh`` (default) - :class:`ZenohTransport`
- ``iot``  - :class:`IotMqttTransport`
- ``bridge``  - :class:`BridgeTransport` (Zenoh + IoT)

Subsequent calls in the same process bump the refcount but do NOT switch
backends. To change the backend, every consumer must release first
(``release_transport`` until refcount is 0) and then a new selection is made.

:mod:`strands_robots.mesh.session` owns the legacy zenoh path independently
but delegates every non-zenoh call to this factory, and reads the backend
variable through :func:`_select_backend` so both modules accept the same set
and report an unknown value once.
"""

from __future__ import annotations

import logging
import os
import threading

from strands_robots.mesh.transport.base import MeshTransport

logger = logging.getLogger(__name__)


_TRANSPORT: MeshTransport | None = None
_TRANSPORT_REFS: int = 0
_TRANSPORT_BACKEND: str = ""
_LOCK = threading.Lock()

#: Backends :func:`_construct` can build. Kept beside that dispatch so a new
#: transport grows the accepted set and the constructor in one edit.
_VALID_BACKENDS: tuple[str, ...] = ("zenoh", "iot", "bridge")

#: Distinct unknown ``STRANDS_MESH_BACKEND`` values already reported. Latched
#: per value rather than once per process: :mod:`strands_robots.mesh.session`
#: reads the variable through :func:`_select_backend` on every publish, so an
#: unlatched report would emit one line per message, while a single flag would
#: swallow a second, different typo that the operator has equally not been told
#: about. Tests clear this set.
_REPORTED_UNKNOWN_BACKENDS: set[str] = set()


def unknown_backend_message(raw: str) -> str:
    """Build the report for an unrecognised ``STRANDS_MESH_BACKEND`` value.

    Both readers of that variable log this: :func:`_select_backend` here, and
    :func:`strands_robots.mesh.session._backend_choice`, which is the one
    :class:`strands_robots.mesh.Mesh` consults on every publish. One builder
    rather than a copy in each module because the wording carries the whole
    substitution -- a peer that asked for a cloud transport comes up on the LAN
    default while every other signal looks healthy -- so a reader told less than
    the other is a reader told nothing useful.

    Args:
        raw: The value as read from the environment, unnormalized. The report
            quotes it verbatim: a normalized whitespace-only value renders as
            ``''``, which reads as unset, and unset is a different situation
            (it falls back to ``zenoh`` by design and correctly says nothing).

    Returns:
        The formatted report, naming the variable, the accepted set and the
        consequence of the substitution.
    """
    return (
        f"Unknown STRANDS_MESH_BACKEND={raw!r} - falling back to 'zenoh'. "
        f"Valid backends: {', '.join(_VALID_BACKENDS)}. "
        "This peer joins the LAN mesh only; a cloud subscriber will not receive from it."
    )


def _select_backend() -> str:
    """Resolve ``STRANDS_MESH_BACKEND``, defaulting to ``zenoh``.

    This resolves the value for transport construction.
    :func:`strands_robots.mesh.session._backend_choice` reads the same variable
    to decide whether to delegate to a transport at all, and both report through
    :func:`unknown_backend_message`, so neither reader can tell an operator less
    than the other.

    Acceptance is case- and whitespace-insensitive, but the report quotes the
    **raw** value: the normalized form renders a variable set to whitespace as
    ``''``, which reads as unset, and unset is a different situation (it falls
    back to ``zenoh`` by design and correctly says nothing).

    An unknown value falls back to ``zenoh`` rather than raising - a typo must
    not stop a robot from joining any mesh at all - and is reported once per
    distinct raw value. The report is the only channel that carries the
    substitution: a peer asked for a cloud transport comes up on the LAN default
    and every other signal looks healthy, so the message has to name the
    variable, the accepted set and the consequence to be actionable.

    Returns:
        One of :data:`_VALID_BACKENDS`.
    """
    raw = os.getenv("STRANDS_MESH_BACKEND", "zenoh")
    normalized = raw.strip().lower()
    if normalized in _VALID_BACKENDS:
        return normalized
    if raw not in _REPORTED_UNKNOWN_BACKENDS:
        _REPORTED_UNKNOWN_BACKENDS.add(raw)
        logger.warning("%s", unknown_backend_message(raw))
    return "zenoh"


def _construct(backend: str) -> MeshTransport:
    """Build a fresh transport for *backend*.

    Imports are deferred (inside this function) to avoid import-time
    circular dependencies: factory → zenoh_transport → session → factory.
    """
    if backend == "iot":
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        return IotMqttTransport()
    if backend == "bridge":
        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        return BridgeTransport()

    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

    return ZenohTransport()


def get_transport() -> MeshTransport | None:
    """Acquire (or reuse) the process-wide transport singleton.

    Increments the refcount each call. Returns ``None`` if the underlying
    backend's :meth:`connect` failed (Zenoh missing, certs missing, broker
    unreachable, etc.) - callers MUST treat ``None`` the same way they
    treated ``get_session() is None`` historically: skip mesh activity and
    move on without raising.
    """
    global _TRANSPORT, _TRANSPORT_REFS, _TRANSPORT_BACKEND  # noqa: PLW0603
    with _LOCK:
        if _TRANSPORT is not None:
            _TRANSPORT_REFS += 1
            return _TRANSPORT

        backend = _select_backend()
        transport = _construct(backend)

        # Try to connect; bail out and keep the singleton None on failure.
        ok = transport.connect()  # type: ignore[attr-defined]
        if not ok:
            logger.debug(
                "[mesh.transport] %s backend connect failed - staying off",
                backend,
            )
            return None

        _TRANSPORT = transport
        _TRANSPORT_REFS = 1
        _TRANSPORT_BACKEND = backend
        logger.info("[mesh.transport] %s backend ready", backend)
        return _TRANSPORT


def release_transport() -> None:
    """Release one reference. Closes when the last is gone. Idempotent."""
    global _TRANSPORT, _TRANSPORT_REFS, _TRANSPORT_BACKEND  # noqa: PLW0603
    with _LOCK:
        if _TRANSPORT_REFS <= 0:
            return
        _TRANSPORT_REFS -= 1
        if _TRANSPORT_REFS <= 0 and _TRANSPORT is not None:
            try:
                _TRANSPORT.close()
            except Exception:
                pass
            _TRANSPORT = None
            _TRANSPORT_REFS = 0
            _TRANSPORT_BACKEND = ""


def current_transport() -> MeshTransport | None:
    """Return the transport without bumping the refcount, or ``None``."""
    with _LOCK:
        return _TRANSPORT


def current_backend() -> str:
    """Return the backend name (``"zenoh"`` / ``"iot"`` / ``""`` if not running)."""
    with _LOCK:
        return _TRANSPORT_BACKEND
