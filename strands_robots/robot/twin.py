"""A hardware Robot's twin: the simulated body in the default world that mirrors what it measures and commands."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot import Robot


def mirror(robot: Robot, targets: np.ndarray, *, step: bool = False, measured: bool = False) -> None:
    """Write ``targets`` to this robot's twin (if any), under the twin's world lock.

    ``step=True`` (what ``act`` passes) then advances the twin's world one control step, as a simulated
    ``act`` does; ``manual_time()`` still suppresses it. ``measured=True`` (what ``observe`` passes) also
    places the twin's joints AT the measurement, so it shows the real pose with no step.
    """
    twin = robot.twin
    if twin is not None and twin.sim is not None:
        with twin.sim.lock:
            if measured:
                with contextlib.suppress(NotImplementedError):  # an engine that cannot place joints: command only
                    twin.sim.backend.set_joints(twin.id, targets)
            twin.body.write(targets)
        if step:
            twin.sim.control_step()
