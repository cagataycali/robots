"""The real arm's joints, read and never written, for the twin to follow.

An operator moves the physical SO-101 by hand and expects the twin on the page
to follow. :class:`BusMirror` is the reader that makes that true: one thread
opens the servo bus, reads ``Present_Position`` from every motor at ~20 Hz, and
publishes the latest reading. The sim session fed by it does not step physics -
it writes those angles into ``qpos`` and runs the kinematics, so the geometry
the browser draws is the real arm's pose.

What this module must never do is write a servo register. lerobot's bus
``connect(handshake=True)`` pings motors and reads their firmware; its
``disconnect()`` **disables torque by default**, which is a write - it is
called with ``disable_torque=False`` here. Nothing else on the bus is touched:
no ``configure``, no ``Torque_Enable``, no ``Goal_Position``. Torque stays in
whatever state the operator left it, and the reader says so.

Failure is closed: a port that does not exist, is held by another process, or
answers with the wrong motors puts the mirror in ``error`` with the reason,
and the session shows that instead of a stale pose.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_HZ = 20.0
#: STS3215: 4096 ticks per turn, 2048 at the centre of travel.
_TICKS_PER_TURN = 4096
_CENTRE = 2048
#: A reading older than this is stale: the reader thread is stuck or the bus went away.
STALE_AFTER = 0.5


def ticks_to_rad(ticks: int) -> float:
    """A raw STS3215 position as radians about the centre of travel.

    This is the estimate for an arm lerobot has no calibration file for: the
    servo's 2048 is taken as the model's zero. It is right up to the offset a
    calibration would record, which is why the snapshot labels it ``estimate``.
    """
    return (int(ticks) - _CENTRE) * (2.0 * math.pi / _TICKS_PER_TURN)


@dataclass(frozen=True)
class Reading:
    """One sweep of the bus: raw ticks per motor name and when it was read."""

    ticks: Mapping[str, int]
    at: float  # time.monotonic()

    def age(self, now: float | None = None) -> float:
        """Seconds since this sweep, on the monotonic clock."""
        return (time.monotonic() if now is None else now) - self.at


class BusMirror:
    """Read-only joint positions from a Feetech servo bus, in its own thread.

    Args:
        port: The serial device, e.g. ``/dev/cu.usbmodem5AB01818061``.
        motors: ``{name: id}`` in joint order. Defaults to the SO-101's six.
        bus_factory: Builds the bus object (``connect``, ``sync_read``,
            ``disconnect``) - tests pass a fake; the default builds lerobot's
            ``FeetechMotorsBus`` for the given motors with no calibration.
    """

    def __init__(
        self,
        port: str,
        motors: Mapping[str, int] | None = None,
        *,
        bus_factory: Callable[[str, Mapping[str, int]], Any] | None = None,
    ):
        self.port = port
        self.motors = dict(motors or SO101_MOTORS)
        self._factory = bus_factory or _lerobot_bus
        self._lock = threading.Lock()
        self._reading: Reading | None = None
        self._error: str | None = None
        self._connected = False
        self._hz = 0.0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"mirror-{port.rsplit('/', 1)[-1]}", daemon=True)
        self._thread.start()

    # -- readers (any thread) ------------------------------------------------

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until the bus is open or failed to open."""
        return self._ready.wait(timeout)

    @property
    def error(self) -> str | None:
        """Why the bus is not answering, or None while it is."""
        with self._lock:
            return self._error

    @property
    def reading(self) -> Reading | None:
        """The latest sweep, or None before the first."""
        with self._lock:
            return self._reading

    def qpos(self) -> tuple[float, ...] | None:
        """Every motor's angle in radians, in ``motors`` order - or None when there is no fresh reading."""
        reading = self.reading
        if reading is None or reading.age() > STALE_AFTER:
            return None
        return tuple(ticks_to_rad(reading.ticks[name]) for name in self.motors)

    def health(self) -> dict[str, Any]:
        """What the snapshot carries about the bus: honest about what is not known."""
        with self._lock:
            reading, error, connected, hz = self._reading, self._error, self._connected, self._hz
        return {
            "port": self.port,
            "connected": connected,
            "hz": round(hz, 1),
            "age_ms": None if reading is None else round(reading.age() * 1000),
            "ticks": None if reading is None else dict(reading.ticks),
            "angles": "estimate: (ticks - 2048) * 2pi / 4096, no calibration applied",
            "writes": "none - torque is whatever the operator left it",
            "error": error,
        }

    def close(self) -> None:
        """Stop reading and release the port. Never writes."""
        self._stop.set()
        self._thread.join(5.0)

    # -- worker --------------------------------------------------------------

    def _run(self) -> None:
        bus = None
        try:
            bus = self._factory(self.port, self.motors)
            bus.connect(handshake=True)
            with self._lock:
                self._connected = True
        except Exception as exc:
            self._fail(f"could not open {self.port}: {type(exc).__name__}: {exc}")
            self._ready.set()
            return
        self._ready.set()
        window, count = time.monotonic(), 0
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                raw = bus.sync_read("Present_Position", normalize=False)
                ticks = {name: int(raw[name]) for name in self.motors}
                count += 1
                with self._lock:
                    self._reading = Reading(ticks, time.monotonic())
                    if t0 - window >= 0.5:
                        self._hz = count / (t0 - window)
                        window, count = t0, 0
                time.sleep(max(0.0, 1.0 / _HZ - (time.monotonic() - t0)))
        except Exception as exc:
            self._fail(f"lost {self.port}: {type(exc).__name__}: {exc}")
        finally:
            try:
                # disable_torque=False: the default would WRITE Torque_Enable=0 to every motor.
                bus.disconnect(disable_torque=False)
            except Exception:
                logger.debug("mirror: bus close failed", exc_info=True)
            with self._lock:
                self._connected = False

    def _fail(self, reason: str) -> None:
        logger.warning("mirror: %s", reason)
        with self._lock:
            self._error = reason
            self._connected = False


#: The SO-101's motors in the order the model's joints ``1``..``6`` are declared.
SO101_MOTORS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def _lerobot_bus(port: str, motors: Mapping[str, int]) -> Any:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors={name: Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100) for name, mid in motors.items()},
        calibration=None,
    )
