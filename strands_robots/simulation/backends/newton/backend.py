"""The MuJoCo world with MuJoCo-Warp as its solver: built, read and rendered by ``MuJoCoBackend``, stepped on the GPU.

``num_envs`` copies of the world step together; env 0 is mirrored into the CPU ``MjData`` (lazily, on the first
read after a step) so every read, render and contact query keeps MuJoCo's meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....types import JointState, Pose
from ...physics import Contact, Physics
from ...scene import Scene
from ..mujoco import MuJoCoBackend
from .stepper import NewtonStepper


class NewtonBackend(MuJoCoBackend):
    def __init__(self, **cfg: Any) -> None:
        super().__init__(**cfg)
        self._stepper = NewtonStepper(self.device, self.num_envs)
        self._stale = False

    def _fresh(self) -> None:
        if self._stale and self.data is not None:
            self._stepper.sync_to(self.data)
            self._stale = False

    def load(self, scene: Scene) -> None:
        self._fresh()
        super().load(scene)
        self._stepper.load(self.model, self.data)

    def step(self, n: int) -> None:
        self._stepper.step(n)
        self._stale = self._stale or n > 0

    def reset(self, keyframe: str | None) -> None:
        super().reset(keyframe)
        self._stale = False
        self._stepper.sync_from(self.data)

    def time(self) -> float:
        self._fresh()
        return super().time()

    def joint_state(self, body: str) -> JointState:
        self._fresh()
        return super().joint_state(body)

    def set_targets(self, body: str, targets: np.ndarray) -> None:
        super().set_targets(body, targets)
        self._stepper.set_ctrl(self._index[body].actuator_ids, targets)

    def body_pose(self, body: str) -> Pose:
        self._fresh()
        return super().body_pose(body)

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        self._fresh()
        super().set_gains(body, kp, kd)
        self._stepper.load(self.model, self.data)  # the device model is a copy: re-upload the edited gains

    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None:
        self._fresh()
        super().set_pose(body, pose, zero_velocity=zero_velocity)
        self._stepper.sync_from(self.data)

    def render(self, camera: str) -> np.ndarray:
        self._fresh()
        return super().render(camera)

    def contacts(self) -> list[Contact]:
        self._fresh()
        return super().contacts()

    def physics(self) -> Physics:
        assert self.model is not None
        return Physics("newton", float(self.model.opt.timestep), 1)

    def close(self) -> None:
        self._stepper.close()
        super().close()
