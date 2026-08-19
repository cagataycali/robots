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
the reads collided continuously and every joint feature in the dashboard drew
nothing: no joint bars, no history traces, no motion detection. The SDK's three
retries did not help, because all three landed inside the same collision.

The lock lives on the DEVICE, not on any one caller, because that is what is
actually being shared: the mesh modules, the teleop rail and the dashboard all
hold different wrappers around the same lerobot robot. It is an ``RLock`` so a
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


def bus_lock(device: Any) -> threading.RLock:
    """Return the one lock guarding ``device``'s bus, creating it on first use.

    Args:
        device: The object whose bus is being shared -- normally a lerobot
            robot, or anything else with ``get_observation``/``send_action``.

    Returns:
        The device's lock. Every caller passing the same device gets the same
        lock object, including callers in other modules.
    """
    existing = getattr(device, _LOCK_ATTR, None)
    if existing is not None:
        return existing  # type: ignore[return-value]

    with _registry_guard:
        # Re-read inside the guard: another thread may have just created it.
        existing = getattr(device, _LOCK_ATTR, None)
        if existing is not None:
            return existing  # type: ignore[return-value]
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
        return device.get_observation()


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
        return device.send_action(action)
