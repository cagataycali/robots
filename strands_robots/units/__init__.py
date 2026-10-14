"""Units: a policy declares what it speaks (``policy.units``), the robot's description converts.

    conv = Units.for_robot(robot.description, "lerobot")
    q_policy = conv.from_rad(obs.joints.positions)     # what the policy should see
    q_rad    = conv.to_rad(policy_action)              # what Robot.act should receive

``Robot.run(policy, units="auto")`` does this around every ``act``; ``units="rad"`` disables it; a
scheme name forces it when the policy cannot say.
"""

from __future__ import annotations

from . import lerobot, rad  # noqa: F401  (register the schemes)
from .scheme import SCHEMES, Units, declared_units, register, resolve_units

__all__ = ["Units", "SCHEMES", "register", "resolve_units", "declared_units"]
