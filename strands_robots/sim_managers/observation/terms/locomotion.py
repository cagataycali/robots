"""First-class locomotion observation terms.

Each term returns a 1-D ``float64`` vector that the :class:`ObservationManager`
concatenates (after optional per-term scale + clip) into the policy observation.
These mirror the canonical Isaac Lab / RSL-RL locomotion observation group.
"""

from __future__ import annotations

from typing import Any

from strands_robots.sim_managers.base import EnvState, FloatArray, Term, register_term


@register_term("observation", "base_lin_vel")
class BaseLinVel(Term):
    """Base-frame linear velocity ``[vx, vy, vz]`` (shape ``(3,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return state.base_lin_vel


@register_term("observation", "base_ang_vel")
class BaseAngVel(Term):
    """Base-frame angular velocity ``[wx, wy, wz]`` (shape ``(3,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return state.base_ang_vel


@register_term("observation", "projected_gravity")
class ProjectedGravity(Term):
    """Gravity unit vector projected into the base frame (shape ``(3,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return state.projected_gravity


@register_term("observation", "joint_pos")
class JointPos(Term):
    """Joint positions relative to the default pose (``joint_pos - default``).

    Relative encoding is the locomotion convention: it centres the observation
    on the nominal stance so the policy sees deviations, not absolute angles.
    """

    def __call__(self, state: EnvState) -> FloatArray:
        assert state.default_joint_pos is not None  # set in EnvState.__post_init__
        return state.joint_pos - state.default_joint_pos


@register_term("observation", "joint_vel")
class JointVel(Term):
    """Joint velocities (shape ``(n_joints,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return state.joint_vel


@register_term("observation", "last_action")
class LastAction(Term):
    """Previous control action (shape ``(n_actions,)``)."""

    def __call__(self, state: EnvState) -> FloatArray:
        return state.last_action


@register_term("observation", "velocity_commands")
class VelocityCommands(Term):
    """The active base-velocity command vector the policy should track.

    Args:
        command_name: Key of the command in :attr:`EnvState.commands`.
    """

    def __init__(self, command_name: str = "base_velocity", **params: Any) -> None:
        super().__init__(command_name=command_name, **params)
        self.command_name = command_name

    def __call__(self, state: EnvState) -> FloatArray:
        return state.command(self.command_name)
