"""Units: the robot speaks radians; a policy speaks whatever its dataset stored.

A LeRobot checkpoint un-normalises to the units of the recordings it was trained on — for a real SO-10x that
is the Feetech bus range (body joints −100…100, gripper 0…100), for a MuJoCo-recorded dataset it is radians.
Nothing here guesses: a policy *declares* its units (``policy.units``), the robot's description converts.

    conv = Units.for_robot(robot.description, "lerobot_so10x")
    q_policy = conv.from_rad(obs.joints.positions)     # what the policy should see
    q_rad    = conv.to_rad(policy_action)              # what Robot.act should receive

``Robot.run(policy, units="auto")`` does exactly that around every ``act``; ``units="rad"`` disables it;
a name forces it when the policy cannot say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .description import Description

RAD = "rad"
LEROBOT_SO10X = "lerobot_so10x"
KNOWN = (RAD, LEROBOT_SO10X)


@dataclass(frozen=True)
class Units:
    """An affine map rad -> policy units per action key: ``u = lo_u + (q - lo) / (hi - lo) * (hi_u - lo_u)``."""

    name: str
    keys: tuple[str, ...]
    lo: np.ndarray  # rad
    hi: np.ndarray  # rad
    lo_u: np.ndarray  # policy units
    hi_u: np.ndarray  # policy units

    @classmethod
    def for_robot(cls, description: Description, name: str) -> Units:
        """The converter between this description's radians and the named unit system."""
        keys = description.action_keys
        if name == RAD:
            lo = np.array([description.limits.get(k, (-np.pi, np.pi))[0] for k in keys])
            hi = np.array([description.limits.get(k, (-np.pi, np.pi))[1] for k in keys])
            return cls(RAD, keys, lo, hi, lo.copy(), hi.copy())
        if name == LEROBOT_SO10X:
            missing = [k for k in keys if k not in description.limits]
            if missing:
                raise ValueError(
                    f"{name!r} needs joint limits for every action key; {description.name} lacks {missing}"
                )
            lo = np.array([description.limits[k][0] for k in keys])
            hi = np.array([description.limits[k][1] for k in keys])
            grip = np.array([_is_gripper(k, description) for k in keys])
            return cls(name, keys, lo, hi, np.where(grip, 0.0, -100.0), np.where(grip, 100.0, 100.0))
        raise ValueError(f"unknown units {name!r}; known: {', '.join(KNOWN)}")

    @property
    def is_identity(self) -> bool:
        return self.name == RAD

    def from_rad(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        frac = (q - self.lo) / (self.hi - self.lo)
        return self.lo_u + frac * (self.hi_u - self.lo_u)

    def to_rad(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        frac = (u - self.lo_u) / (self.hi_u - self.lo_u)
        return self.lo + frac * (self.hi - self.lo)


def _is_gripper(key: str, description: Description) -> bool:
    if key == "gripper":
        return True
    acts = description.gripper.get("actuators") if description.gripper else None
    if acts and key in description.action_keys:
        idx = description.action_keys.index(key)
        return str(idx + 1) in {str(a) for a in acts} or key in {str(a) for a in acts}
    return False


def declared_units(policy: Any) -> str | None:
    """What a policy says it speaks: ``policy.units`` -> a name in KNOWN, or None when it cannot say."""
    u = getattr(policy, "units", None)
    if u is None:
        return None
    if u not in KNOWN:
        raise ValueError(f"policy declares units {u!r}; known: {', '.join(KNOWN)}")
    return str(u)


def resolve_units(policy: Any, requested: str | None, description: Description) -> Units | None:
    """``requested`` = "auto" (use the policy's declaration, radians if it has none), "rad", a known name, or None."""
    if requested is None or requested == RAD:
        return None
    if requested == "auto":
        name = declared_units(policy)
        if name is None or name == RAD:
            return None
        return Units.for_robot(description, name)
    return Units.for_robot(description, requested)
