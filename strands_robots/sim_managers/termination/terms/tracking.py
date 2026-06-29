"""Whole-body-tracking (WBT) termination terms."""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, Term, register_term
from strands_robots.sim_managers.motion import MOTION_TARGET_POS, read_motion_target


@register_term("termination", "motion_divergence")
class MotionDivergence(Term):
    """Terminate (failure) when the pose strays too far from the reference.

    Fires when the L2 norm of the per-joint position error exceeds
    ``threshold`` - the standard "lost the motion" early-termination used in
    imitation / tracking training. This is a failure, not a timeout.

    Args:
        threshold: Maximum allowed L2 joint-position error (radians).
    """

    def __init__(self, threshold: float = 1.5, **params: Any) -> None:
        super().__init__(threshold=threshold, **params)
        self.threshold = float(threshold)

    def __call__(self, state: EnvState) -> bool:
        target = np.asarray(read_motion_target(state, MOTION_TARGET_POS), dtype=np.float64)
        if target.shape != state.joint_pos.shape:
            raise ValueError(
                f"motion target length {target.shape[0]} != joint count {state.num_joints}; "
                "the motion_clip frames must have one column per joint."
            )
        return float(np.linalg.norm(state.joint_pos - target)) > self.threshold
