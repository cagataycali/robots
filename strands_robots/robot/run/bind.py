"""Bind a policy to a robot (gains, units, requirements, reset), then decide one action per step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...policy import PolicyLike, as_policy
from ...types import Observation
from ...units import Units, resolve_units

if TYPE_CHECKING:
    from ..robot import Robot


@dataclass(frozen=True)
class Bound:
    """A policy made ready for a robot: the adapted policy and the unit converter (or None for radians)."""

    policy: PolicyLike
    units: Units | None


def bind(robot: Robot, policy: PolicyLike | Callable[[Observation], Any], units: str | None) -> Bound:
    """What every loop does before step one: the policy's gains, its units, its requirements, a reset."""
    pol = as_policy(policy)
    attach = getattr(pol, "attach", None)
    if callable(attach):  # connect / load / admit against THIS robot, before units and requirements
        attach(robot)
    gains = getattr(pol, "gains", None)
    if gains is not None and robot.sim is not None:  # the policy's PD is part of the policy, not of the world
        robot.set_gains(*gains)
    conv = resolve_units(pol, units, robot.description)
    req = getattr(pol, "requirements", None)
    if req is not None:
        req().check(robot.observe(), who=repr(pol))
    pol.reset()
    return Bound(pol, conv)


def release(bound: Bound) -> None:
    """After the last step: let the policy close what ``attach`` opened (a session, a server)."""
    detach = getattr(bound.policy, "detach", None)
    if callable(detach):
        detach()


def decide(robot: Robot, bound: Bound, obs: Observation, instruction: str | None) -> Any:
    """One inference in the policy's units, returned as an action in the robot's radians."""
    pol, conv = bound.policy, bound.units
    if conv is None:
        return robot.coerce(pol.act(obs, instruction=instruction))
    seen = obs.with_joints(conv.from_rad(obs.joints.positions))
    return robot.coerce(conv.to_rad(np.asarray(pol.act(seen, instruction=instruction), dtype=float)))
