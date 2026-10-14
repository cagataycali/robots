"""The contract a physics world implements, and the registry of engines."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import numpy as np

from ..types import JointState, Pose
from .physics import Contact, Physics
from .scene import Scene

_BACKENDS: dict[str, str] = {
    "mujoco": "strands_robots.simulation.backends.mujoco:MuJoCoBackend",
}


class Backend(ABC):
    """The thirteen methods a physics world implements. Nothing else is engine-specific."""

    def __init__(
        self,
        *,
        physics: str = "native",
        headless: bool = True,
        device: str | None = None,
        num_envs: int = 1,
        render_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.physics_name = physics
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.render_size = render_size

    @abstractmethod
    def load(self, scene: Scene) -> None: ...  # (re)build the world; keep joint state of surviving bodies

    @abstractmethod
    def step(self, n: int) -> None: ...

    @abstractmethod
    def reset(self, keyframe: str | None) -> None: ...

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def joint_state(self, body: str) -> JointState: ...

    @abstractmethod
    def set_targets(self, body: str, targets: np.ndarray) -> None: ...

    @abstractmethod
    def body_pose(self, body: str) -> Pose: ...

    @abstractmethod
    # teleport; a fixed body's new pose is already recorded in the scene, so the backend rebuilds it there
    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None: ...

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        """Actuator PD gains for one robot (``action_keys`` order). Engines that cannot say so."""
        raise NotImplementedError(f"{type(self).__name__} cannot change actuator gains at runtime")

    @abstractmethod
    def render(self, camera: str) -> np.ndarray: ...

    @abstractmethod
    def contacts(self) -> list[Contact]: ...

    @abstractmethod
    def physics(self) -> Physics: ...

    @abstractmethod
    def close(self) -> None: ...


def load_backend(name: str) -> type[Backend]:
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown simulation backend {name!r}; choose from {sorted(_BACKENDS)}") from None
    mod_name, cls_name = target.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)
