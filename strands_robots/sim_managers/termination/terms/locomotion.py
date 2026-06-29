"""First-class locomotion termination terms.

A termination term returns ``bool`` (terminate this step or not). Terms whose
``is_time_out`` class attribute is ``True`` signal episode truncation (a
horizon timeout), which an RL trainer treats differently from a genuine failure
(it bootstraps the value function across a timeout but not a failure).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, Term, register_term


@register_term("termination", "time_out")
class TimeOut(Term):
    """Truncate when the episode reaches its horizon (``step_count >= max``)."""

    is_time_out = True

    def __call__(self, state: EnvState) -> bool:
        return state.step_count >= state.max_episode_length


@register_term("termination", "bad_orientation")
class BadOrientation(Term):
    """Terminate when the base tilts past ``limit_angle`` from upright.

    Args:
        limit_angle: Maximum tilt of the gravity vector from straight-down,
            radians.
    """

    def __init__(self, limit_angle: float = 1.0, **params: Any) -> None:
        super().__init__(limit_angle=limit_angle, **params)
        self.limit_angle = float(limit_angle)

    def __call__(self, state: EnvState) -> bool:
        gravity = state.projected_gravity
        norm = float(np.linalg.norm(gravity))
        if norm == 0.0:
            return False
        # Tilt of -gravity from world up == angle of gravity_z below the horizon.
        cos_tilt = -gravity[2] / norm
        cos_tilt = float(np.clip(cos_tilt, -1.0, 1.0))
        return float(np.arccos(cos_tilt)) > self.limit_angle


@register_term("termination", "base_height_below_threshold")
class BaseHeightBelowThreshold(Term):
    """Terminate when the base falls below ``min_height``.

    Args:
        min_height: Minimum allowed base height, metres.
    """

    def __init__(self, min_height: float = 0.3, **params: Any) -> None:
        super().__init__(min_height=min_height, **params)
        self.min_height = float(min_height)

    def __call__(self, state: EnvState) -> bool:
        return state.base_height < self.min_height


@register_term("termination", "joint_pos_limit")
class JointPosLimit(Term):
    """Terminate when any joint position exceeds its hard limits.

    Returns ``False`` when ``joint_pos_limits`` is unavailable.
    """

    def __call__(self, state: EnvState) -> bool:
        if state.joint_pos_limits is None:
            return False
        lower = state.joint_pos_limits[:, 0]
        upper = state.joint_pos_limits[:, 1]
        return bool(np.any(state.joint_pos < lower) or np.any(state.joint_pos > upper))
