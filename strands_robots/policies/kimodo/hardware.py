"""Hardware bridge between :class:`KimodoPolicy` and lerobot's ``unitree_g1`` driver.

Kimodo emits per-frame joint targets keyed by URDF joint names
(``left_hip_pitch_joint``, ...); the lerobot :class:`UnitreeG1` hardware driver
expects action keys in its ``G1_29_JointIndex`` enum naming (``kLeftHipPitch.q``,
...). Both sides expose EXACTLY the same 29 joints in the same canonical order,
so the bridge is a pure NAME RENAME — no reordering, no scaling, no units
translation.

This module holds:

* :data:`KIMODO_TO_LEROBOT_G1_JOINTS` - the frozen name map, generated once at
  import time so a lerobot rename surfaces here at import (not silently mid
  rollout).
* :func:`kimodo_action_to_lerobot_g1` - the per-tick action-dict rename.
* :func:`build_lerobot_g1_action_dict` - wrapper the hardware run loop uses:
  takes the policy's raw action-dict and returns an action-dict lerobot's
  ``UnitreeG1.send_action`` accepts verbatim.

The rename is one-way (Kimodo -> hardware). The reverse (observation
name-mapping) is handled by lerobot's driver's ``get_observation`` which
already surfaces ``motor.q`` keys, so no inverse map is needed on the read
path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)


def _build_joint_map() -> dict[str, str]:
    """Build the Kimodo-URDF -> lerobot-enum joint name map.

    Imports lerobot lazily so this module is importable WITHOUT ``lerobot``
    installed (the map is only needed on hardware — sim never calls this).

    Raises:
        ImportError: If lerobot's ``unitree_g1`` subpackage is not available.
            Callers who reach the hardware path but skipped the ``[lerobot]``
            extra get a clear install hint instead of a mysterious KeyError
            deep in :meth:`UnitreeG1.send_action`.
        RuntimeError: If the two joint lists have different lengths (a lerobot
            or Kimodo joint-set change that would silently corrupt the map).
    """
    try:
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "Kimodo hardware bridge for the Unitree G1 requires lerobot's "
            "``unitree_g1`` driver. Install with:\n"
            "  pip install 'strands-robots[lerobot,unitree]'\n"
            "or drop the hardware target and stay in sim."
        ) from exc

    lerobot_names = [f"{j.name}.q" for j in G1_29_JointIndex]
    if len(lerobot_names) != len(KIMODO_G1_JOINTS):
        raise RuntimeError(
            "Kimodo <-> lerobot G1 joint count mismatch: Kimodo emits "
            f"{len(KIMODO_G1_JOINTS)} joints, lerobot expects "
            f"{len(lerobot_names)}. One of them changed its DOF set; the "
            "rename table in strands_robots.policies.kimodo.hardware must be "
            "audited before you drive the real robot."
        )
    return dict(zip(KIMODO_G1_JOINTS, lerobot_names, strict=True))


#: Frozen at import time. ``left_hip_pitch_joint`` -> ``kLeftHipPitch.q`` etc.
#: Access lazily: not built until first use so ``import strands_robots.policies.kimodo``
#: does not pull in lerobot for pure-sim users. Set on first call.
_KIMODO_TO_LEROBOT: dict[str, str] | None = None


def get_joint_map() -> dict[str, str]:
    """Return the Kimodo-URDF -> lerobot-enum joint name map, building it on first call."""
    global _KIMODO_TO_LEROBOT
    if _KIMODO_TO_LEROBOT is None:
        _KIMODO_TO_LEROBOT = _build_joint_map()
    return _KIMODO_TO_LEROBOT


def kimodo_action_to_lerobot_g1(
    kimodo_action: dict[str, float],
) -> dict[str, float]:
    """Rename a Kimodo joint-dict to the lerobot ``UnitreeG1`` action-dict.

    Args:
        kimodo_action: The single per-tick dict returned by
            :meth:`KimodoPolicy.get_actions` (keys are
            :data:`KIMODO_G1_JOINTS`, values are radians).

    Returns:
        A new dict with lerobot-enum keys (``kLeftHipPitch.q``, ...) suitable
        for :meth:`lerobot.robots.unitree_g1.UnitreeG1.send_action`. Extra
        keys in the input (e.g. base pose leftovers) are dropped rather than
        silently forwarded — Kimodo's output shape is well-defined and the
        driver rejects unknown keys.

    Raises:
        KeyError: If any of the 29 canonical Kimodo joints is absent from
            ``kimodo_action`` (a policy contract violation).
    """
    joint_map = get_joint_map()
    missing = [k for k in joint_map if k not in kimodo_action]
    if missing:
        raise KeyError(
            "Kimodo action-dict is missing required G1 joints: "
            f"{missing}.\nKimodoPolicy.get_actions() must return all "
            f"{len(joint_map)} joints; got keys={sorted(kimodo_action)}."
        )
    return {joint_map[k]: float(kimodo_action[k]) for k in joint_map}


def build_lerobot_g1_action_dict(
    kimodo_action: dict[str, float],
    *,
    extra_action_keys: dict[str, float] | None = None,
) -> dict[str, float]:
    """Full hardware-side action-dict for one control tick.

    This is what :class:`~strands_robots.hardware_robot.Robot` calls between
    ``policy.get_actions`` and ``self.robot.send_action`` when the resolved
    policy is Kimodo and the robot is a Unitree G1.

    Args:
        kimodo_action: Per-tick output of :meth:`KimodoPolicy.get_actions`.
        extra_action_keys: Optional overrides / additions merged AFTER the
            joint rename (locomotion controller remote inputs, etc.). Rare —
            most callers pass ``None``.

    Returns:
        Action-dict ready for :meth:`UnitreeG1.send_action`.
    """
    action = kimodo_action_to_lerobot_g1(kimodo_action)
    if extra_action_keys:
        action.update(extra_action_keys)
    return action


__all__ = [
    "get_joint_map",
    "kimodo_action_to_lerobot_g1",
    "build_lerobot_g1_action_dict",
]
