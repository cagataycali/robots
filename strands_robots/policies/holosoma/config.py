"""Configuration for the Holosoma locomotion provider.

Every number here is read from Amazon FAR's ``holosoma_inference`` package at
commit ``bccd4d7`` (2026-09-04): the ``loco-g1-29dof`` observation preset
(``config/config_values/observation.py``), the ``locomotion`` task preset
(``config/config_values/task.py``: ``rl_rate=50``, ``policy_action_scale=0.25``,
``gait_period=1.0``, ``desired_base_height=0.75``) and the ``g1-29dof`` robot
preset (``config/config_values/robot.py``: ``default_dof_angles``). The PD gains
are NOT here on purpose: upstream reads them from the ONNX metadata
(``policies/base.py`` ``_resolve_control_gains``: "config override > ONNX
metadata > error") and so does :class:`~strands_robots.policies.holosoma.HolosomaPolicy`;
``kps`` / ``kds`` below are the optional override slot.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

# Joint count of the Unitree G1 29-DOF model the released checkpoints drive.
G1_NUM_JOINTS = 29

#: Width of the ``actor_obs`` input of every released locomotion checkpoint:
#: actions(29) + base_ang_vel(3) + command_ang_vel(1) + command_lin_vel(2) +
#: cos_phase(2) + dof_pos(29) + dof_vel(29) + projected_gravity(3) + sin_phase(2).
HOLOSOMA_OBS_DIM = 100

#: Nominal stance, ``dof_names`` order (legs L/R, waist, left arm, right arm).
#: ``holosoma_inference/config/config_values/robot.py:75-81``.
HOLOSOMA_G1_DEFAULT_ANGLES: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    0.0, 0.0, 0.0,  # waist yaw / roll / pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
)  # fmt: skip

#: Per-term observation scales of the ``loco-g1-29dof`` preset
#: (``config_values/observation.py:48-59``). Terms absent here scale by 1.
HOLOSOMA_G1_OBS_SCALES: dict[str, float] = {
    "base_ang_vel": 0.25,
    "dof_vel": 0.05,
}

#: Command ranges the released G1 checkpoints were trained on (ONNX metadata
#: ``command_ranges``). Used as the fallback when a checkpoint carries none.
HOLOSOMA_G1_COMMAND_RANGES: dict[str, tuple[float, float]] = {
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-1.0, 1.0),
    "ang_vel_yaw": (-1.0, 1.0),
}


@dataclass(frozen=True)
class HolosomaConfig:
    """Numbers that shape one Holosoma locomotion rollout.

    Attributes:
        algorithm: Which released checkpoint family to fetch when
            ``checkpoint`` names no file: ``"fastsac"`` (upstream's safety
            default and lerobot's default) or ``"ppo"``.
        num_actions: Joints the network drives. The released G1 checkpoints
            output 29; this is also what the MuJoCo torque shim reads to
            decide how many actuators it flips to torque mode.
        n_obs_joints: Joints the ``dof_pos`` / ``dof_vel`` blocks observe
            (29, the whole body).
        action_scale: Multiplier on the clipped network output before it is
            added to ``default_angles`` (upstream ``policy_action_scale``).
        action_clip: Symmetric clip on the raw network output (upstream
            ``np.clip(policy_action, -100, 100)``).
        rl_rate: Control rate in Hz the gait clock integrates at (upstream
            ``rl_rate``). The MuJoCo shim steps physics at 0.005 s x 4.
        gait_period: Seconds per full gait cycle (upstream ``gait_period``).
        height_cmd: Base height in metres the shim places the pelvis at on
            install (upstream ``desired_base_height``). Not an observation.
        default_angles: Nominal stance in ``dof_names`` order, 29 values.
        obs_scales: Per-term scales; see :data:`HOLOSOMA_G1_OBS_SCALES`.
        command_ranges: Per-slot ``(lo, hi)`` the velocity command is clipped
            to, overridden by the checkpoint's own metadata when present.
        kps / kds: Optional per-joint gain override (29 each). Empty means
            "use the ONNX metadata", which is what upstream does by default.
    """

    algorithm: str = "fastsac"
    num_actions: int = G1_NUM_JOINTS
    n_obs_joints: int = G1_NUM_JOINTS
    action_scale: float = 0.25
    action_clip: float = 100.0
    rl_rate: float = 50.0
    gait_period: float = 1.0
    height_cmd: float = 0.75
    default_angles: tuple[float, ...] = HOLOSOMA_G1_DEFAULT_ANGLES
    obs_scales: dict[str, float] = field(default_factory=lambda: dict(HOLOSOMA_G1_OBS_SCALES))
    command_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(HOLOSOMA_G1_COMMAND_RANGES))
    kps: tuple[float, ...] = ()
    kds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm not in ("fastsac", "ppo"):
            raise ValueError(f"HolosomaConfig.algorithm must be 'fastsac' or 'ppo', got {self.algorithm!r}")
        if self.num_actions != G1_NUM_JOINTS or self.n_obs_joints != G1_NUM_JOINTS:
            raise ValueError(
                f"HolosomaConfig drives the Unitree G1 29-DOF model: num_actions and n_obs_joints must be "
                f"{G1_NUM_JOINTS}, got {self.num_actions} / {self.n_obs_joints}. A different humanoid needs "
                "its own checkpoint family, joint table and gains."
            )
        if len(self.default_angles) != G1_NUM_JOINTS:
            raise ValueError(
                f"HolosomaConfig.default_angles must have {G1_NUM_JOINTS} entries, got {len(self.default_angles)}"
            )
        for name, gains in (("kps", self.kps), ("kds", self.kds)):
            if gains and len(gains) != G1_NUM_JOINTS:
                raise ValueError(
                    f"HolosomaConfig.{name} must be empty or have {G1_NUM_JOINTS} entries, got {len(gains)}"
                )
        for name in ("action_scale", "rl_rate", "gait_period", "height_cmd", "action_clip"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"HolosomaConfig.{name} must be a positive finite number, got {value!r}")

    @property
    def phase_dt(self) -> float:
        """Radians the gait phase advances per control tick: ``2*pi / (rl_rate * gait_period)``."""
        return 2.0 * math.pi / (self.rl_rate * self.gait_period)

    def with_gains(self, kps: tuple[float, ...], kds: tuple[float, ...]) -> HolosomaConfig:
        """Return a copy carrying the gains read from a checkpoint's metadata."""
        return dataclasses.replace(self, kps=tuple(float(k) for k in kps), kds=tuple(float(k) for k in kds))


__all__ = [
    "G1_NUM_JOINTS",
    "HOLOSOMA_G1_COMMAND_RANGES",
    "HOLOSOMA_G1_DEFAULT_ANGLES",
    "HOLOSOMA_G1_OBS_SCALES",
    "HOLOSOMA_OBS_DIM",
    "HolosomaConfig",
]
