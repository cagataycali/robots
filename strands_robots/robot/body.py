"""A Body is something with joints you can read and targets you can write.

``SimBody`` is a handle into a Simulation; a ``Driver`` (robot.drivers) is the hardware Body. Every
loop is written once against ``Body`` and never knows which it has.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ..types import JointState, Pose

if TYPE_CHECKING:
    from ..simulation import Simulation


@runtime_checkable
class Body(Protocol):
    """Something with joints you can read and targets you can write."""

    names: tuple[str, ...]

    def read(self) -> JointState: ...

    def write(self, targets: np.ndarray) -> None: ...

    def pose(self) -> Pose | None: ...

    def images(self) -> Mapping[str, np.ndarray]: ...

    def extras(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class SimBody:
    """A handle into a Simulation: the body named ``id`` in ``sim``."""

    def __init__(self, sim: Simulation, id: str, names: tuple[str, ...]) -> None:
        self.sim = sim
        self.id = id
        self.names = names

    def read(self) -> JointState:
        return self.sim.backend.joint_state(self.id)

    def write(self, targets: np.ndarray) -> None:
        self.sim.backend.set_targets(self.id, targets)

    def pose(self) -> Pose | None:
        return self.sim.backend.body_pose(self.id)

    def images(self) -> Mapping[str, np.ndarray]:
        return {c.name: self.sim.backend.render(c.name) for c in self.sim.scene.cameras if c.attach == self.id}

    def extras(self) -> Mapping[str, Any]:
        return {}

    def close(self) -> None:
        pass
