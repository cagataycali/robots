"""Whole-body-tracking (WBT) reward terms.

Tracking terms return a bounded ``(0, 1]`` Gaussian kernel of the per-joint
imitation error, matching the ``track_*_exp`` convention of the locomotion
terms: the :class:`~strands_robots.sim_managers.RewardManager` applies the
configured ``weight``. They read the reference targets published by the
``motion_clip`` command term.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, Term, register_term
from strands_robots.sim_managers.motion import (
    MOTION_TARGET_POS,
    MOTION_TARGET_VEL,
    read_motion_target,
)


def _tracking_error(value: np.ndarray, target: np.ndarray, quantity: str) -> float:
    """Sum of squared per-joint error, validating matching shapes."""
    target = np.asarray(target, dtype=np.float64)
    if target.shape != value.shape:
        raise ValueError(
            f"motion {quantity} target length {target.shape[0]} != joint count {value.shape[0]}; "
            "the motion_clip frames must have one column per joint."
        )
    return float(np.sum((value - target) ** 2))


@register_term("reward", "track_joint_pos_exp")
class TrackJointPosExp(Term):
    """Reward matching the reference joint pose (Gaussian kernel).

    Args:
        std: Tracking tolerance; smaller is stricter.
    """

    def __init__(self, std: float = 0.5, **params: Any) -> None:
        super().__init__(std=std, **params)
        self.std = float(std)

    def __call__(self, state: EnvState) -> float:
        target = read_motion_target(state, MOTION_TARGET_POS)
        err = _tracking_error(state.joint_pos, target, "position")
        return float(np.exp(-err / self.std**2))


@register_term("reward", "track_joint_vel_exp")
class TrackJointVelExp(Term):
    """Reward matching the reference joint velocity (Gaussian kernel).

    Args:
        std: Tracking tolerance; smaller is stricter.
    """

    def __init__(self, std: float = 1.0, **params: Any) -> None:
        super().__init__(std=std, **params)
        self.std = float(std)

    def __call__(self, state: EnvState) -> float:
        target = read_motion_target(state, MOTION_TARGET_VEL)
        err = _tracking_error(state.joint_vel, target, "velocity")
        return float(np.exp(-err / self.std**2))
