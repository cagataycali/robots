"""A hardware Robot's twin: the simulated body in the default world that mirrors what it measures and commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot import Robot


def mirror(robot: Robot, targets: np.ndarray) -> None:
    """Write ``targets`` to this robot's twin (if any), under the twin's world lock."""
    twin = robot.twin
    if twin is not None and twin.sim is not None:
        with twin.sim.lock:
            twin.body.write(targets)
