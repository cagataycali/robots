"""Unit adapter between a robot's joint positions and the FLUX 3 Action SO-101 convention.

FLUX 3 Action (Black Forest Labs, 2026-09-22) was trained on ``lerobot/community_dataset_v3``
SO-101 episodes, where the five arm joints are absolute angles in DEGREES (the LeRobot
``use_degrees`` calibration) and the gripper is an opening PERCENT in ``[0, 100]``. The
MuJoCo ``so101`` model in strands-robots exposes joint positions in RADIANS and its gripper
joint travels ``[-0.1745, 1.7453]`` rad (``so101_new_calib.xml``). Real SO-101 hardware
through the lerobot driver reports degrees + percent already.

The conversion is pure arithmetic, deterministic, and explicit; no unit is guessed from
value magnitude. Both directions live here so the round trip is testable without a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Canonical SO-101 joint order, identical in lerobot, the MuJoCo model (joints "1".."6")
# and the FLUX 3 Action checkpoint (``dataset_statistics.json`` names).
SO101_JOINT_LABELS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Gripper joint travel of ``so101_new_calib.xml`` (MuJoCo, radians): closed .. open.
SO101_SIM_GRIPPER_RANGE_RAD: tuple[float, float] = (-0.17453292519943295, 1.7453292519943295)

# Degree offsets that carry the MuJoCo ``so101_new_calib`` zero (arm stretched out
# horizontally, every joint at 0) into the frame the released checkpoint was
# normalised in. FLUX 3 Action was trained with PER-DATASET quantile normalisation
# (``configs/so101/community_corpus.json`` in black-forest-labs/flux-action), so the
# absolute numbers only mean something relative to the checkpoint's ``state`` q01/q99
# window: pan [-92, 54], lift [85, 184], elbow [61, 179], wrist_flex [-24, 81],
# roll [-125, 60], gripper [0.5, 53.6]. Those bounds are the classic LeRobot SO-100/101
# calibration where "zero" is the arm pointing straight up and the folded rest pose
# reads roughly (0, 190, 180, 70, 0, 0). In MuJoCo the same folded pose is
# lift ~ +100 deg and elbow ~ +90 deg, hence +90 on both.
SO101_SIM_JOINT_OFFSETS_DEG: tuple[float, float, float, float, float] = (0.0, 90.0, 90.0, 0.0, 0.0)

_DEG_PER_RAD = 180.0 / math.pi


@dataclass(frozen=True)
class UnitAdapter:
    """Convert between robot joint units and FLUX 3 Action units (deg + gripper percent).

    Args:
        joint_units: ``"rad"`` when the robot reports radians (MuJoCo), ``"deg"`` when it
            already reports degrees (lerobot hardware with ``use_degrees``).
        joint_signs: per-arm-joint direction (+1 or -1) applied before the offset, for a
            model whose joint axis points the other way. Five values.
        joint_offsets_deg: additive per-arm-joint offset in degrees applied AFTER the
            rad->deg conversion and the sign, so a model whose zero differs from the
            checkpoint's calibration zero can be aligned explicitly. Five values (arm
            joints only). Defaults to :data:`SO101_SIM_JOINT_OFFSETS_DEG`.
        gripper_range: ``(closed, open)`` in robot units; maps linearly onto
            ``[0, 100]`` percent. ``(0, 100)`` is the identity for lerobot hardware.
    """

    joint_units: str = "rad"
    joint_signs: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets_deg: tuple[float, float, float, float, float] = SO101_SIM_JOINT_OFFSETS_DEG
    gripper_range: tuple[float, float] = SO101_SIM_GRIPPER_RANGE_RAD

    def __post_init__(self) -> None:
        if self.joint_units not in ("rad", "deg"):
            raise ValueError(f"joint_units must be 'rad' or 'deg', got {self.joint_units!r}")
        if len(self.joint_offsets_deg) != 5:
            raise ValueError(f"joint_offsets_deg needs 5 values (arm joints), got {len(self.joint_offsets_deg)}")
        if len(self.joint_signs) != 5 or any(s not in (1.0, -1.0) for s in self.joint_signs):
            raise ValueError(f"joint_signs needs 5 values of +1 or -1, got {self.joint_signs!r}")
        lo, hi = self.gripper_range
        if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
            raise ValueError(f"gripper_range must be two distinct finite values, got {self.gripper_range!r}")

    @property
    def _scale(self) -> float:
        return _DEG_PER_RAD if self.joint_units == "rad" else 1.0

    def robot_to_model(self, joints: Sequence[float]) -> list[float]:
        """Robot joint positions (6, robot units) -> FLUX 3 Action state (deg x5 + percent)."""
        if len(joints) != 6:
            raise ValueError(f"expected 6 joint values [{', '.join(SO101_JOINT_LABELS)}], got {len(joints)}")
        arm = [
            float(q) * self._scale * sign + off
            for q, sign, off in zip(joints[:5], self.joint_signs, self.joint_offsets_deg, strict=True)
        ]
        lo, hi = self.gripper_range
        pct = (float(joints[5]) - lo) / (hi - lo) * 100.0
        return [*arm, pct]

    def model_to_robot(self, action: Sequence[float]) -> list[float]:
        """FLUX 3 Action output (deg x5 + percent) -> robot joint targets (6, robot units)."""
        if len(action) != 6:
            raise ValueError(f"expected 6 action values, got {len(action)}")
        arm = [
            (float(a) - off) * sign / self._scale
            for a, sign, off in zip(action[:5], self.joint_signs, self.joint_offsets_deg, strict=True)
        ]
        lo, hi = self.gripper_range
        q = lo + float(action[5]) / 100.0 * (hi - lo)
        return [*arm, q]


__all__ = ["SO101_JOINT_LABELS", "SO101_SIM_GRIPPER_RANGE_RAD", "SO101_SIM_JOINT_OFFSETS_DEG", "UnitAdapter"]
