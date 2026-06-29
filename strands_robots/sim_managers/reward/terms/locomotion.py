"""First-class locomotion reward terms.

Each term returns a non-negative scalar *magnitude*; the :class:`RewardManager`
applies the configured ``weight`` (whose sign distinguishes a reward from a
penalty, following the Isaac Lab / RSL-RL convention). Tracking terms return a
bounded ``(0, 1]`` Gaussian kernel; penalties return a squared magnitude.

The base-velocity command is read from ``state.command(command_name)`` and is
expected to be ``[vx, vy, wz]`` (planar linear + yaw rate).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, Term, register_term


@register_term("reward", "track_lin_vel_xy_exp")
class TrackLinVelXYExp(Term):
    """Reward tracking of the commanded planar linear velocity (Gaussian kernel).

    Args:
        std: Tracking tolerance; smaller is stricter.
        command_name: Command key holding ``[vx, vy, wz]``.
    """

    def __init__(self, std: float = 0.25, command_name: str = "base_velocity", **params: Any) -> None:
        super().__init__(std=std, command_name=command_name, **params)
        self.std = float(std)
        self.command_name = command_name

    def __call__(self, state: EnvState) -> float:
        cmd = state.command(self.command_name)
        err = np.sum((cmd[:2] - state.base_lin_vel[:2]) ** 2)
        return float(np.exp(-err / self.std**2))


@register_term("reward", "track_ang_vel_z_exp")
class TrackAngVelZExp(Term):
    """Reward tracking of the commanded yaw rate (Gaussian kernel).

    Args:
        std: Tracking tolerance; smaller is stricter.
        command_name: Command key holding ``[vx, vy, wz]``.
    """

    def __init__(self, std: float = 0.25, command_name: str = "base_velocity", **params: Any) -> None:
        super().__init__(std=std, command_name=command_name, **params)
        self.std = float(std)
        self.command_name = command_name

    def __call__(self, state: EnvState) -> float:
        cmd = state.command(self.command_name)
        err = (cmd[2] - state.base_ang_vel[2]) ** 2
        return float(np.exp(-err / self.std**2))


@register_term("reward", "lin_vel_z_l2")
class LinVelZL2(Term):
    """Penalise vertical base velocity (squared)."""

    def __call__(self, state: EnvState) -> float:
        return float(state.base_lin_vel[2] ** 2)


@register_term("reward", "ang_vel_xy_l2")
class AngVelXYL2(Term):
    """Penalise roll/pitch angular velocity (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        return float(np.sum(state.base_ang_vel[:2] ** 2))


@register_term("reward", "flat_orientation_l2")
class FlatOrientationL2(Term):
    """Penalise non-flat base orientation via projected-gravity xy (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        return float(np.sum(state.projected_gravity[:2] ** 2))


@register_term("reward", "orientation_l2")
class OrientationL2(Term):
    """Penalise deviation of projected gravity z from the upright value (-1)."""

    def __call__(self, state: EnvState) -> float:
        return float((state.projected_gravity[2] + 1.0) ** 2)


@register_term("reward", "dof_torques_l2")
class DofTorquesL2(Term):
    """Penalise applied joint torques (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        assert state.joint_torque is not None
        return float(np.sum(state.joint_torque**2))


@register_term("reward", "dof_acc_l2")
class DofAccL2(Term):
    """Penalise joint accelerations (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        assert state.joint_acc is not None
        return float(np.sum(state.joint_acc**2))


@register_term("reward", "dof_vel_l2")
class DofVelL2(Term):
    """Penalise joint velocities (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        return float(np.sum(state.joint_vel**2))


@register_term("reward", "action_rate_l2")
class ActionRateL2(Term):
    """Penalise the change in action between steps (sum of squares)."""

    def __call__(self, state: EnvState) -> float:
        return float(np.sum((state.action - state.last_action) ** 2))


@register_term("reward", "joint_pos_limits")
class JointPosLimits(Term):
    """Penalise joint positions outside their soft limits (sum of violations).

    Uses ``joint_pos_limits`` shrunk to ``soft_joint_pos_limit_factor`` of the
    hard range about its midpoint. Zero when limits are unavailable.
    """

    def __call__(self, state: EnvState) -> float:
        if state.joint_pos_limits is None:
            return 0.0
        lower = state.joint_pos_limits[:, 0]
        upper = state.joint_pos_limits[:, 1]
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) * state.soft_joint_pos_limit_factor
        soft_lower = mid - half
        soft_upper = mid + half
        below = np.clip(soft_lower - state.joint_pos, a_min=0.0, a_max=None)
        above = np.clip(state.joint_pos - soft_upper, a_min=0.0, a_max=None)
        return float(np.sum(below + above))


@register_term("reward", "joint_vel_limits")
class JointVelLimits(Term):
    """Penalise joint velocities exceeding a fraction of their limit.

    Args:
        soft_ratio: Fraction of the velocity limit treated as the threshold.
    """

    def __init__(self, soft_ratio: float = 1.0, **params: Any) -> None:
        super().__init__(soft_ratio=soft_ratio, **params)
        self.soft_ratio = float(soft_ratio)

    def __call__(self, state: EnvState) -> float:
        if state.joint_vel_limits is None:
            return 0.0
        threshold = state.joint_vel_limits * self.soft_ratio
        excess = np.clip(np.abs(state.joint_vel) - threshold, a_min=0.0, a_max=None)
        return float(np.sum(excess))


@register_term("reward", "feet_air_time")
class FeetAirTime(Term):
    """Reward foot air time at touchdown to encourage stepping, not shuffling.

    Rewards ``(air_time - threshold)`` on the step a foot makes first contact,
    gated to only apply while a non-trivial velocity command is active. Needs
    ``feet_air_time`` and ``feet_contact``; returns 0 if either is absent.

    Args:
        threshold: Target air time per step, seconds.
        command_name: Command key gating the reward (norm > 0.1).
    """

    def __init__(self, threshold: float = 0.4, command_name: str = "base_velocity", **params: Any) -> None:
        super().__init__(threshold=threshold, command_name=command_name, **params)
        self.threshold = float(threshold)
        self.command_name = command_name
        self._prev_contact: np.ndarray | None = None

    def reset(self, state: EnvState | None = None, *, rng: np.random.Generator | None = None) -> None:
        self._prev_contact = None

    def __call__(self, state: EnvState) -> float:
        if state.feet_air_time is None or state.feet_contact is None:
            return 0.0
        contact = state.feet_contact
        if self._prev_contact is None:
            self._prev_contact = np.zeros_like(contact)
        first_contact = np.logical_and(contact, np.logical_not(self._prev_contact))
        reward = float(np.sum((state.feet_air_time - self.threshold) * first_contact))
        self._prev_contact = contact.copy()
        cmd = state.commands.get(self.command_name)
        if cmd is not None and float(np.linalg.norm(cmd[:2])) <= 0.1:
            return 0.0
        return reward


@register_term("reward", "feet_slide")
class FeetSlide(Term):
    """Penalise foot sliding: foot speed while in contact (sum of squares).

    Reads per-foot planar speed from ``state.extras['feet_lin_vel']`` (shape
    ``(n_feet,)`` or ``(n_feet, 2/3)``). Returns 0 if unavailable.
    """

    def __call__(self, state: EnvState) -> float:
        if state.feet_contact is None:
            return 0.0
        feet_vel = state.extras.get("feet_lin_vel")
        if feet_vel is None:
            return 0.0
        vel = np.asarray(feet_vel, dtype=np.float64)
        speed_sq = vel**2 if vel.ndim == 1 else np.sum(vel**2, axis=-1)
        return float(np.sum(speed_sq * state.feet_contact))


@register_term("reward", "alive")
class Alive(Term):
    """Constant per-step survival reward (1.0)."""

    def __call__(self, state: EnvState) -> float:
        return 1.0


@register_term("reward", "termination_penalty")
class TerminationPenalty(Term):
    """Penalise non-timeout episode termination (1.0 when terminated)."""

    def __call__(self, state: EnvState) -> float:
        return 1.0 if state.terminated else 0.0
