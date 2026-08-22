"""One reader at a time on a robot's motor bus.

A serial motor bus is a single conversation: the host writes a sync-read packet
addressed to a set of servo ids, then reads the replies off the wire. Two
threads doing that at once interleave their packets, and the feetech/dynamixel
SDKs refuse outright::

    ConnectionError("Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4,
    5, 6] after 3 tries. [TxRxResult] Port is in use!")

Which is what a strands mesh peer did to itself. Four independent threads in one
child process reach for the same device -- the state probe that fills the fleet
snapshot, the hardware camera publisher (which calls ``get_observation()`` at
``STRANDS_MESH_CAMERA_HZ``, and lerobot's ``get_observation()`` reads the MOTORS
before it grabs any frame), the sensors probe, and the IoT camera offload -- and
teleop writes to it besides. Nobody was serialising them, so on real hardware
the reads collided continuously and every joint consumer drew nothing: no
positions in the fleet snapshot, no history traces, no motion detection. The
SDK's three retries did not help: all three landed inside the same collision.

The lock lives on the DEVICE, not on any one caller, because that is what is
actually being shared: the mesh modules, the teleop rail and any application on
top of them all hold different wrappers around the same lerobot robot. It is an ``RLock`` so a
caller that already holds it can read again without deadlocking (lerobot's own
``get_observation()`` is free to call back into a locked read).

This module deliberately knows nothing about lerobot or the mesh: it is import-
safe from anywhere, which is the only way every reader can be made to share one
lock without an import cycle.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: Attribute the per-device lock is cached under. Named for us, so a driver's
#: own attributes can never be mistaken for it.
_LOCK_ATTR = "_strands_bus_lock"

#: Guards lock CREATION, so two threads racing to be first cannot end up with a
#: lock each -- which would serialise nothing at all.
_registry_guard = threading.Lock()

#: Used when a device refuses attribute assignment (``__slots__``, a frozen
#: dataclass, a proxy). Shared by all such devices: over-serialising two
#: unrelated robots is slow, and letting them collide is broken.
_fallback_lock = threading.RLock()


def _cached_lock(device: Any) -> threading.RLock | None:
    """The lock already cached on ``device``, or ``None`` on first use.

    One typed read of an untyped ``getattr``: the attribute is named for this
    module, so nothing but a lock created here is ever stored under it.

    Args:
        device: The object whose bus lock is being looked up.

    Returns:
        The cached lock, or ``None`` when the device has not been locked yet.
    """
    lock: threading.RLock | None = getattr(device, _LOCK_ATTR, None)
    return lock


def bus_lock(device: Any) -> threading.RLock:
    """Return the one lock guarding ``device``'s bus, creating it on first use.

    Args:
        device: The object whose bus is being shared -- normally a lerobot
            robot, or anything else with ``get_observation``/``send_action``.

    Returns:
        The device's lock. Every caller passing the same device gets the same
        lock object, including callers in other modules.
    """
    existing = _cached_lock(device)
    if existing is not None:
        return existing

    with _registry_guard:
        # Re-read inside the guard: another thread may have just created it.
        existing = _cached_lock(device)
        if existing is not None:
            return existing
        lock = threading.RLock()
        try:
            setattr(device, _LOCK_ATTR, lock)
        except Exception:  # noqa: BLE001 - __slots__, frozen, or a proxy
            logger.debug(
                "%s will not hold a bus lock; falling back to the shared one",
                type(device).__name__,
            )
            return _fallback_lock
        return lock


def read_observation(device: Any) -> Any:
    """Read one observation from ``device`` with exclusive use of its bus.

    The one call every reader should use. Blocking is the point: a probe that
    waits its turn produces a reading, while a probe that barges in produces a
    ``Port is in use!`` and no data for anyone.

    Args:
        device: The robot to read. Must expose ``get_observation()``.

    Returns:
        Whatever the driver's ``get_observation()`` returns.

    Raises:
        Exception: Anything the driver raises, unchanged -- callers already
            handle hardware errors, and the lock is released either way.
    """
    with bus_lock(device):
        return _read_recovering_a_stale_flag(device, getattr(device, "bus", None), lambda: device.get_observation())


def write_action(device: Any, action: Any) -> Any:
    """Send one action to ``device`` with exclusive use of its bus.

    Takes the SAME lock as :func:`read_observation`, because a write that
    interleaves with a read corrupts both halves of the exchange -- teleop
    moving an arm while a probe reads its position is the common case.

    Args:
        device: The robot to command. Must expose ``send_action()``.
        action: The action, in whatever shape the driver accepts.

    Returns:
        Whatever the driver's ``send_action()`` returns.
    """
    with bus_lock(device):
        try:
            return device.send_action(action)
        except Exception as exc:  # noqa: BLE001 - re-raised, only better explained
            if port_busy_action(exc, already_recovered=False) != "clear_and_retry":
                raise
            raise ConnectionError(write_refusal(device, exc)) from exc


#: Register every SO-101-family bus reports positions from. Named here so the
#: joints-only read is one obvious call and not a string buried in a probe.
_POSITION_REGISTER = "Present_Position"


def _num_read_retries(device: Any) -> int | None:
    cfg = getattr(device, "config", None)
    n = getattr(cfg, "num_read_retries", None)
    return n if isinstance(n, int) else None


def read_joints(device: Any) -> Any:
    """Read ONLY the joint positions, so a broken camera cannot hide them.

    Measured on real hardware: an arm published ZERO joints for eleven hours
    while its mesh presence stayed healthy and non-stale. One line at startup
    explained it -- ``state probe 'hw_joints' failed ...
    RuntimeError('OpenCVCamera(1) read failed')`` -- and then the log went quiet.

    The cause is in lerobot's ``get_observation()``: it sync-reads the motors
    FIRST, then loops over the cameras calling ``read_latest()``. A camera that
    raises therefore throws away the joint positions ALREADY IN HAND. Every
    joints consumer went through that call -- the fleet snapshot, the joint
    history traces, motion detection, and teleop's 30Hz publisher -- so a single
    dead USB camera silently disarmed an entire arm.

    Joints and frames are independent facts about a robot and must fail
    independently. When the driver exposes its motor bus we read that directly
    (under the SAME lock as everything else, or this reintroduces the collision
    this module exists to prevent). When it does not, we fall back to the full
    observation: a driver whose only reader is ``get_observation`` cannot be
    asked for less, and pretending otherwise would return nothing at all.

    Args:
        device: The robot to read. ``device.bus.sync_read`` is used when present.

    Returns:
        A mapping of ``"<motor>.pos"`` -> position, shaped exactly like the joint
        half of ``get_observation()`` so callers need no new branch. On the
        fallback path, whatever ``get_observation()`` returns (frames included).

    Raises:
        Exception: Anything the driver raises, unchanged.
    """
    bus = getattr(device, "bus", None)
    if bus is None or not hasattr(bus, "sync_read"):
        return read_observation(device)
    retries = _num_read_retries(device)

    def _sync_read() -> Any:
        if retries is None:
            return bus.sync_read(_POSITION_REGISTER)
        try:
            return bus.sync_read(_POSITION_REGISTER, num_retry=retries)
        except TypeError:
            # A bus implementation without the retry keyword: the read still
            # matters more than the retry policy.
            return bus.sync_read(_POSITION_REGISTER)

    with bus_lock(device):
        raw = _read_recovering_a_stale_flag(device, bus, _sync_read)
    if not isinstance(raw, dict):
        return raw
    # ``.pos`` is lerobot's own suffix (see SOFollower.get_observation) and the
    # shape the rest of this codebase already parses.
    return {f"{motor}.pos": value for motor, value in raw.items()}


_PORT_BUSY_SIGNATURE = "Port is in use!"


def port_busy_action(error: Any, already_recovered: bool) -> str:
    """Decide what a port-busy failure means, given whether we already recovered once.

    Measured on real hardware: both arms published ZERO joints for hours with healthy presence,
    because one read had failed with ``[TxRxResult] Port is in use!`` and
    every read after it failed identically. The mechanism is in the vendored SDK: ``txPacket`` sets
    ``port.is_using = True``, and the callers clear it with a bare ``port.is_using = False`` AFTER
    the call - there is no ``try``/``finally`` anywhere in that file. So an exception, a timeout or a
    cancellation between those two lines leaves the flag set for the LIFE OF THE PROCESS, and the
    arm is mute until someone notices and respawns it. Retrying cannot help: the flag is checked
    before the port is touched.

    Clearing that flag is safe only under a proof, never on a hunch, because clearing a flag a
    concurrent exchange genuinely holds would interleave two conversations on one UART and hand back
    positions that were never measured. The proof this module can offer is its own lock: every read
    and write here happens inside :func:`bus_lock`, so while we hold it a port that still claims to
    be in use is either stale (its owner died mid-exchange) or held by a reader that bypasses the
    lock. The first is recoverable; the second is a bug that must be SAID OUT LOUD, not smoothed
    over. Hence: clear once, and if the very next read is busy again, report a real owner - the
    second failure is the evidence that distinguishes them.
    """
    if _PORT_BUSY_SIGNATURE not in str(error):
        return "reraise"
    return "report_real_owner" if already_recovered else "clear_and_retry"


def clear_stale_port_busy(bus: Any) -> bool:
    """Clear the SDK's in-use flag, only if it is actually set. Returns whether it was.

    Deliberately narrow: it touches one boolean on the port handler and nothing else - no reopen, no
    reconnect, no write to a motor. A bus that does not expose a port handler (a stub, a sim, a
    future driver) is left alone and reported as unrecovered, so the caller re-raises the original
    error rather than claiming a repair it did not make.
    """
    handler = getattr(bus, "port_handler", None)
    if handler is None or not getattr(handler, "is_using", False):
        return False
    handler.is_using = False
    return True


def _read_recovering_a_stale_flag(device: Any, bus: Any, do_read: Any) -> Any:
    """Run a READ, and if the port is merely marked in-use by a dead exchange, clear it once.

    Must be called with :func:`bus_lock` already held - that lock IS the proof (see
    :func:`port_busy_action`). Shared by every read path here so a stranded flag cannot mute one
    caller while another recovers; a read is the only operation that can do this safely, because
    re-reading a position changes nothing in the world.
    """
    try:
        return do_read()
    except Exception as exc:  # noqa: BLE001 - re-raised unless provably recoverable
        if port_busy_action(exc, already_recovered=False) != "clear_and_retry":
            raise
        if not clear_stale_port_busy(bus):
            raise
        count = _record_recovery(device, bus)
        logger.warning(
            "%s: the motor bus was left marked in-use by an exchange that never finished; cleared "
            "that flag while holding this arm's bus lock and read again (%d time(s) this session). "
            "Nothing else can recover it, and every later read would have failed the same way.",
            getattr(device, "name", None) or type(device).__name__,
            count,
        )
        try:
            return do_read()
        except Exception as second:  # noqa: BLE001
            if port_busy_action(second, already_recovered=True) == "report_real_owner":
                raise ConnectionError(
                    "the motor bus reports in-use again immediately after that flag was cleared, "
                    "while this process holds the bus lock - so the port has a REAL owner outside "
                    "bus_access (another process, or a reader that skips the lock). Ask the OS who "
                    f"holds it: /usr/sbin/lsof {getattr(bus, 'port', '')} ({second})"
                ) from second
            raise


def write_refusal(device: Any, error: Any) -> str:
    """Explain a port-busy WRITE, which this module deliberately does not recover.

    A read may clear a stranded in-use flag (see :func:`port_busy_action`): re-reading a position
    changes nothing in the world, so the worst case of being wrong is a bad number, and the retry is
    bounded at one. A WRITE is not symmetrical, and the asymmetry is the point:

    * The exchange that stranded the flag may itself have been a write. Part of a goal-position packet
      may already be on the wire, so the motors' commanded target is UNKNOWN - and the first thing to
      do with an arm in an unknown commanded state is READ it, not send it another target.
    * Re-sending an action is motion. This module cannot know whether the action in hand is still the
      one the operator wants, or a frame from a teleop stream that has since moved on; replaying a
      stale target is exactly how an arm jumps.

    So the write says what happened, and points at the operation that legitimately clears the flag -
    any read, which the state probe performs every cycle anyway. That makes recovery automatic within
    about a second of telemetry WITHOUT this module ever choosing to move a real arm.
    """
    name = getattr(device, "name", None) or type(device).__name__
    return (
        f"{name}: refusing to re-send this action. The motor bus is marked in-use by an exchange "
        "that never finished, so the arm's commanded position is unknown, and re-sending motion "
        "after an aborted write is not a decision this layer makes. A READ clears that flag safely "
        "(the state probe does it every cycle, so telemetry alone recovers it in about a second) - "
        f"then command the arm again. Original error: {error}"
    )


#: How many stranded in-use flags each port has needed cleared, this process's lifetime.
_RECOVERIES: dict[str, int] = {}
_RECOVERIES_LOCK = threading.Lock()


def _recovery_key(device: Any, bus: Any) -> str:
    """Identify the PORT, because that is what strands - falling back to the device's name."""
    port = getattr(bus, "port", None) or getattr(device, "port", None)
    return str(port or getattr(device, "name", None) or type(device).__name__)


def _record_recovery(device: Any, bus: Any) -> int:
    key = _recovery_key(device, bus)
    with _RECOVERIES_LOCK:
        count = _RECOVERIES.get(key, 0) + 1
        _RECOVERIES[key] = count
    return count


def recovery_count(device: Any, bus: Any = None) -> int:
    """How many times a stranded in-use flag has been cleared for this device's port.

    Exists because the cure is SILENT: from the moment :func:`read_joints` learned to clear a stranded
    flag, an arm that would have gone mute for hours now heals inside one telemetry cycle - which is
    the right behaviour and the wrong amount of information. A flag gets stranded when an exchange
    dies mid-conversation, and the usual reasons for that are physical: a marginal USB cable, a hub
    browning out under load, a connector working loose as the arm moves. Recovering silently would
    hide a degrading rig behind healthy-looking telemetry until the day the recovery itself fails.

    So the count is kept per PORT (what actually strands) and published with the arm's state, where a
    rising number is the evidence a human needs: once is a hiccup, dozens is hardware to replace. It
    is a session counter, not a total - it resets with the process that owns the port, and that is
    honest, because a fresh process cannot know what happened before it.
    """
    if bus is None:
        bus = getattr(device, "bus", None)
    with _RECOVERIES_LOCK:
        return _RECOVERIES.get(_recovery_key(device, bus), 0)
