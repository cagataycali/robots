"""Whole-body-tracking (WBT) observation terms.

These expose the reference-motion targets published by the ``motion_clip``
command term so a tracking policy observes what it is being asked to imitate.
Each reads ``state.extras`` via :func:`~strands_robots.sim_managers.motion.read_motion_target`,
which raises a clear error if no ``motion_clip`` command term ran this step.
"""

from __future__ import annotations

import numpy as np

from strands_robots.sim_managers.base import EnvState, FloatArray, Term, register_term
from strands_robots.sim_managers.motion import (
    MOTION_PHASE,
    MOTION_TARGET_POS,
    MOTION_TARGET_VEL,
    read_motion_target,
)


@register_term("observation", "motion_phase")
class MotionPhase(Term):
    """Cyclic clip phase encoded as ``[sin(2*pi*phase), cos(2*pi*phase)]``.

    The sin/cos encoding is continuous across the loop boundary (phase 1 -> 0),
    unlike the raw scalar, which is the standard phase observation for periodic
    motion tracking.
    """

    def __call__(self, state: EnvState) -> FloatArray:
        phase = float(read_motion_target(state, MOTION_PHASE))
        angle = 2.0 * np.pi * phase
        return np.array([np.sin(angle), np.cos(angle)])


@register_term("observation", "motion_target_joint_pos")
class MotionTargetJointPos(Term):
    """Target joint positions for the current clip frame (shape ``(n_joints,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return np.asarray(read_motion_target(state, MOTION_TARGET_POS), dtype=np.float64)


@register_term("observation", "motion_target_joint_vel")
class MotionTargetJointVel(Term):
    """Target joint velocities for the current clip frame (shape ``(n_joints,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return np.asarray(read_motion_target(state, MOTION_TARGET_VEL), dtype=np.float64)


@register_term("observation", "joint_pos_error")
class JointPosError(Term):
    """Per-joint tracking error ``joint_pos - target`` (shape ``(n_joints,)``).

    Raises:
        ValueError: If the target length does not match the robot's joint count
            (a misconfigured clip), rather than broadcasting silently.
    """

    def __call__(self, state: EnvState) -> FloatArray:
        target = np.asarray(read_motion_target(state, MOTION_TARGET_POS), dtype=np.float64)
        if target.shape != state.joint_pos.shape:
            raise ValueError(
                f"motion target length {target.shape[0]} != joint count {state.num_joints}; "
                "the motion_clip frames must have one column per joint."
            )
        return state.joint_pos - target
