"""The e-stop: the one path that bypasses the safety chain. It latches until an operator approves ``resume``.

``engage`` never takes the world lock or the run-loop lock: it sets the latch, then asks the body for
the fastest stop it has (``Body.estop``). The run loop polls the latch every tick.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from .validators import RequestRefused

if TYPE_CHECKING:
    from ..robot import Robot


class EStopped(RequestRefused):
    """A command sent while the e-stop is latched."""


class Latch:
    """Engaged / released, with who released it and when it was engaged (monotonic)."""

    def __init__(self) -> None:
        self._set = threading.Event()
        self.engaged_at: float | None = None
        self.reason: str | None = None
        self.resumed_by: str | None = None
        self.count = 0  # engagements so far: a run that saw a different count was stopped, even if resumed since

    @property
    def engaged(self) -> bool:
        return self._set.is_set()

    def engage(self, reason: str) -> None:
        if not self._set.is_set():
            self.engaged_at, self.reason = time.monotonic(), reason
            self.count += 1
            self._set.set()

    def release(self, by: str) -> None:
        self.resumed_by = by
        self._set.clear()

    def check(self, robot_id: str) -> None:
        if self._set.is_set():
            raise EStopped(f"{robot_id}: e-stop is engaged ({self.reason}); resume(approved_by=<operator>) first")


def engage(robot: Robot, reason: str = "estop()") -> None:
    """Latch first (so the next tick refuses), then the body's own stop."""
    robot.estop_latch.engage(reason)
    body = robot._body
    if body is not None:
        body.estop()


def guarded_write(robot: Robot, targets: object) -> None:
    """Write under the body's lock, refusing while latched; a stop that lands mid-write is re-applied after it."""
    latch, body = robot.estop_latch, robot.body
    latch.check(robot.id)
    body.write(targets)  # type: ignore[arg-type]
    if latch.engaged:
        body.estop()
        latch.check(robot.id)


def resume(robot: Robot, *, approved_by: str | None) -> None:
    """Release the latch; only a named operator can."""
    if not approved_by or not str(approved_by).strip():
        raise RequestRefused(f"{robot.id}: resume needs approved_by=<the operator who cleared the stop>")
    robot.estop_latch.release(str(approved_by))
