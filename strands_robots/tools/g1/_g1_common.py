"""Shared DDS init and safety helpers for the G1 hardware layer.

Two things need to be true across the driver and (later, issue #358) the
agent tools:

* ``ChannelFactoryInitialize`` is called exactly once per process, with a
  known network interface. Calling it twice from separate imports raises
  from ``unitree_sdk2py``; calling it never leaves subscribers with no bus.
  :func:`ensure_dds` runs it under a lock, records the interface, and is
  idempotent for callers with the same choice.
* A ``ChannelSubscriber`` and a ``ChannelPublisher`` cannot be constructed
  concurrently: the CycloneDDS bindings segfault. :data:`_DDS_INIT_LOCK` is
  the *shared* lock the driver and the tools (issue #358) both hold while
  creating readers or writers. One lock; two consumers.

The ``unitree_sdk2py`` import is lazy - the module attribute stays ``None`` and
is loaded inside :func:`ensure_dds` on first call - so importing this module
without the SDK installed succeeds. That is what lets every test in this repo
mock the bus and skips the SDK entirely on Thor.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error decoder shared with agent tools (issue #358) so the same numeric code
# renders the same text everywhere. Values sourced from ``unitree_sdk2py``
# response codes observed against the real G1; ``0`` is the SDK's success
# marker and always present.
# ---------------------------------------------------------------------------
ERR_CODES: dict[int, str] = {
    0: "OK",
    3102: "RPC_CLIENT_SEND fail",
    3103: "RPC_CLIENT_API_NOT_REG",
    3104: "RPC_CLIENT_API_TIMEOUT",
    7301: "LocoState not available",
    7302: "Invalid FSM id (loco)",
    7303: "Invalid task id (loco)",
    7400: "rt/armsdk topic is occupied",
    7401: "Arm is holding - release first (id=99)",
    7402: "Invalid action id",
    7404: "Invalid FSM id - need FSM in {500, 501, 801}",
}

#: FSM ids where arm-SDK commands are honoured. Outside this set the arm
#: refuses to move at all - so :meth:`~strands_robots.drivers.g1.G1Driver.send_action`
#: checks membership before writing ``rt/armsdk``.
HANDSHAKE_FSMS: frozenset[int] = frozenset({500, 501, 801})

#: FSM ids where locomotion velocity commands are honoured. Narrower than
#: :data:`HANDSHAKE_FSMS` because sitting (500) accepts arm gestures but not
#: walking.
WALK_FSMS: frozenset[int] = frozenset({501, 801})


def decode_code(code: object) -> str:
    """Render a Unitree SDK response code as ``"NNNN (meaning)"``.

    Falls back to a bare stringified value for anything the table does not
    know, so a new error code from a firmware update surfaces at least as its
    integer instead of vanishing into ``"OK"``. Non-integer inputs (``None``,
    the ``ret`` field a mocked client may leave as a string) render as their
    ``repr`` for the same reason: a caller reading the log should see what the
    driver actually received.
    """
    if isinstance(code, int):
        return f"{code} ({ERR_CODES.get(code, 'unknown')})"
    return f"{code!r}"


# ---------------------------------------------------------------------------
# DDS init lock - held by both driver and agent tools (issue #358) around
# every subscriber/publisher construction, because the CycloneDDS bindings
# segfault under concurrent creation. One lock, shared.
# ---------------------------------------------------------------------------
_DDS_INIT_LOCK: threading.Lock = threading.Lock()

# Recorded once by :func:`ensure_dds` for a running process. A second call
# with a different interface is a bug worth catching, not a silent no-op.
_dds_state: dict[str, object] = {"initialized": False, "interface": None}


def ensure_dds(network_interface: str = "eth0") -> str | None:
    """Initialise the DDS channel factory, at most once per process.

    Returns ``None`` on success. Returns a named error string when the SDK is
    missing or the underlying initialise raised - the caller decides whether
    that is fatal (a real bring-up on hardware) or expected (a headless test
    that mocks the bus). The lock covers both the SDK import and the
    ``ChannelFactoryInitialize`` call so two threads racing on the first
    connection cannot double-initialise.

    Args:
        network_interface: The interface to bind CycloneDDS to.

    Returns:
        ``None`` on success. An error string otherwise; never raises.
    """
    with _DDS_INIT_LOCK:
        if _dds_state["initialized"]:
            recorded = _dds_state["interface"]
            if recorded != network_interface:
                # A silent re-bind would attach subscribers to the wrong NIC
                # and produce empty topics with no obvious cause. Refuse.
                return (
                    f"ChannelFactoryInitialize was called with interface "
                    f"{recorded!r}; refusing to re-initialise on {network_interface!r}"
                )
            return None
        try:
            # Lazy import - the SDK is only touched here. On a machine that
            # does not have it (Thor, CI, every unit test) the ImportError is
            # returned as a named string rather than raised, and the caller
            # can decide whether to proceed with a mocked bus.
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return f"unitree_sdk2py is not installed: {exc}"
        try:
            ChannelFactoryInitialize(0, network_interface)
        except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
            message = str(exc).lower()
            if "already" in message or "initialized" in message:
                # Some SDK builds refuse a second init - treat that as success
                # for the interface we recorded.
                _dds_state["initialized"] = True
                _dds_state["interface"] = network_interface
                return None
            return f"ChannelFactoryInitialize failed: {exc}"
        _dds_state["initialized"] = True
        _dds_state["interface"] = network_interface
        logger.info("ChannelFactoryInitialize(0, %r) succeeded", network_interface)
        return None


def reset_dds_state() -> None:
    """Reset the recorded DDS init state - test hook only.

    Nothing in production calls this. The CycloneDDS factory has no
    ``Shutdown`` in ``unitree_sdk2py``, so a real reset requires a fresh
    process; this only lets a test pretend the factory was never initialised.
    """
    with _DDS_INIT_LOCK:
        _dds_state["initialized"] = False
        _dds_state["interface"] = None
