"""``Policy("hold")``: stand still at the pose the run started at, the run-loop form of ``robot.hold()``."""

from __future__ import annotations

import numpy as np

from ..types import Action, Observation
from .base import Policy


class HoldPolicy(Policy):
    """Latches the first observed joints and commands them every step; ``reset`` (each run) re-latches."""

    def __init__(self) -> None:
        self.pose: np.ndarray | None = None

    def reset(self) -> None:
        self.pose = None

    def needs_observation(self, *, instruction: str | None = None) -> bool:
        return False  # joints only: no camera frame is rendered for a hold

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        if self.pose is None:
            self.pose = np.array(obs.joints.positions, dtype=float)
        return self.pose.copy()

    def __repr__(self) -> str:
        return "Policy('hold')"
