"""SONIC decoder constants, transcribed from NVlabs/GR00T-WholeBodyControl.

Every number here is read from
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``
(Apache-2.0) and the deploy loop ``src/g1_deploy_onnx_ref.cpp``; the line
references in the comments point at that revision (main, 2026-09-29). The
decoder they describe is ``model_decoder.onnx`` from the HuggingFace repo
``nvidia/GEAR-SONIC``: input ``obs_dict [1, 994]``, output ``action [1, 29]``.

Two joint orders appear. "Hardware" order is the Unitree motor order, which is
also the Menagerie ``g1.xml`` joint order, the LeRobot dataset
``observation.state`` order and :data:`strands_robots.policies.wbc.WBC_G1_ALL_JOINTS`.
"IsaacLab" order is the order the network was trained in (joints grouped by
depth-first tree traversal). The two permutations below convert between them.
"""

from __future__ import annotations

import math

import numpy as np

# Joint names in hardware order (policy_parameters.hpp comments next to
# g1_action_scale, kps, kds, default_angles). Identical to
# strands_robots.policies.wbc.WBC_G1_ALL_JOINTS; spelled out so this module
# has no import edge into the wbc package.
SONIC_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NUM_JOINTS = 29
TOKEN_DIM = 64
HISTORY_LEN = 10
CONTROL_HZ = 50.0

# policy_parameters.hpp: isaaclab_to_mujoco / mujoco_to_isaaclab.
# hardware index i holds network index ISAACLAB_TO_HARDWARE[i]:
#   q_target_hw[i] = default[i] + action_net[ISAACLAB_TO_HARDWARE[i]] * scale[i]
# network index i holds hardware index HARDWARE_TO_ISAACLAB[i]:
#   body_q_net[i] = q_hw[HARDWARE_TO_ISAACLAB[i]] - default[HARDWARE_TO_ISAACLAB[i]]
ISAACLAB_TO_HARDWARE: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
HARDWARE_TO_ISAACLAB: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

# Motor model (policy_parameters.hpp top): PD gains are derived from the rotor
# armature of each Unitree motor type with a 10 Hz natural frequency and a
# damping ratio of 2. Effort limits feed the per-joint action scale.
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT_LIMIT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_NATURAL_FREQ = 10.0 * 2.0 * math.pi
_DAMPING_RATIO = 2.0


def _stiffness(motor: str) -> float:
    return _ARMATURE[motor] * _NATURAL_FREQ * _NATURAL_FREQ


def _damping(motor: str) -> float:
    return 2.0 * _DAMPING_RATIO * _ARMATURE[motor] * _NATURAL_FREQ


# (motor type, gain multiplier) per joint in hardware order; the multiplier is
# the ``2.0 *`` the header applies to the ankles and to waist roll/pitch.
_MOTOR_PER_JOINT: tuple[tuple[str, float], ...] = (
    ("7520_22", 1.0),  # left_hip_pitch
    ("7520_22", 1.0),  # left_hip_roll
    ("7520_14", 1.0),  # left_hip_yaw
    ("7520_22", 1.0),  # left_knee
    ("5020", 2.0),  # left_ankle_pitch
    ("5020", 2.0),  # left_ankle_roll
    ("7520_22", 1.0),  # right_hip_pitch
    ("7520_22", 1.0),  # right_hip_roll
    ("7520_14", 1.0),  # right_hip_yaw
    ("7520_22", 1.0),  # right_knee
    ("5020", 2.0),  # right_ankle_pitch
    ("5020", 2.0),  # right_ankle_roll
    ("7520_14", 1.0),  # waist_yaw
    ("5020", 2.0),  # waist_roll
    ("5020", 2.0),  # waist_pitch
    ("5020", 1.0),  # left_shoulder_pitch
    ("5020", 1.0),  # left_shoulder_roll
    ("5020", 1.0),  # left_shoulder_yaw
    ("5020", 1.0),  # left_elbow
    ("5020", 1.0),  # left_wrist_roll
    ("4010", 1.0),  # left_wrist_pitch
    ("4010", 1.0),  # left_wrist_yaw
    ("5020", 1.0),  # right_shoulder_pitch
    ("5020", 1.0),  # right_shoulder_roll
    ("5020", 1.0),  # right_shoulder_yaw
    ("5020", 1.0),  # right_elbow
    ("5020", 1.0),  # right_wrist_roll
    ("4010", 1.0),  # right_wrist_pitch
    ("4010", 1.0),  # right_wrist_yaw
)

#: Per-joint proportional gains, hardware order (policy_parameters.hpp ``kps``).
SONIC_KPS: np.ndarray = np.array([_stiffness(m) * k for m, k in _MOTOR_PER_JOINT], dtype=np.float64)
#: Per-joint derivative gains, hardware order (``kds``).
SONIC_KDS: np.ndarray = np.array([_damping(m) * k for m, k in _MOTOR_PER_JOINT], dtype=np.float64)
#: Raw network output -> joint offset scale, hardware order (``g1_action_scale``):
#: ``0.25 * effort_limit / stiffness`` of the joint's motor type (the ankle and
#: waist ``2.0 *`` gain multiplier does not enter the scale, as in the header).
SONIC_ACTION_SCALE: np.ndarray = np.array(
    [0.25 * _EFFORT_LIMIT[m] / _stiffness(m) for m, _ in _MOTOR_PER_JOINT], dtype=np.float64
)
#: Nominal standing pose the network's offsets are added to, hardware order
#: (``default_angles``).
SONIC_DEFAULT_ANGLES: np.ndarray = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # left leg
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # right leg
        0.0,
        0.0,
        0.0,  # waist
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # left arm
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # right arm
    ],
    dtype=np.float64,
)

#: Observation layout of the decoder input, in file order of
#: ``observation_config.yaml`` (g1_deploy_onnx_ref.cpp:1711, 1810-1814):
#: (name, frames, width per frame). Total 994.
OBS_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("token_state", 1, TOKEN_DIM),
    ("his_base_angular_velocity_10frame_step1", HISTORY_LEN, 3),
    ("his_body_joint_positions_10frame_step1", HISTORY_LEN, NUM_JOINTS),
    ("his_body_joint_velocities_10frame_step1", HISTORY_LEN, NUM_JOINTS),
    ("his_last_actions_10frame_step1", HISTORY_LEN, NUM_JOINTS),
    ("his_gravity_dir_10frame_step1", HISTORY_LEN, 3),
)
OBS_DIM = sum(frames * width for _, frames, width in OBS_LAYOUT)

#: Pelvis height the deploy loop and our torque shim seed at install (m).
SONIC_BASE_HEIGHT = 0.79

#: The 64-D token of a stable standing pose for the default SONIC checkpoint
#: (gear_sonic/utils/inference/initial_poses.py, LATENT_INITIAL_MOTION_TOKEN).
#: Checkpoint specific: a different SONIC variant decodes it to a different pose.
STANDING_TOKEN: np.ndarray = np.array(
    [
        -0.0625,
        0.0000,
        -0.0625,
        -0.1250,
        -0.1875,
        -0.0625,
        0.1875,
        0.2500,
        0.1875,
        -0.1250,
        0.0625,
        -0.0625,
        -0.2500,
        -0.2500,
        -0.3125,
        -0.0625,
        0.0000,
        -0.0625,
        -0.1250,
        -0.1875,
        0.0000,
        -0.2500,
        0.0000,
        -0.2500,
        -0.0625,
        0.0625,
        0.1250,
        -0.1250,
        0.2500,
        0.1875,
        0.2500,
        -0.1250,
        0.1250,
        0.1875,
        -0.0625,
        0.0000,
        -0.1875,
        -0.1875,
        0.2500,
        0.0000,
        0.0000,
        -0.1250,
        0.0625,
        0.0000,
        -0.0625,
        -0.0625,
        0.1875,
        -0.0625,
        0.0000,
        0.0625,
        0.1250,
        0.0625,
        0.1250,
        0.0625,
        0.1250,
        0.0000,
        0.1250,
        0.1875,
        0.0000,
        0.0000,
        0.0625,
        0.0625,
        0.1875,
        0.0625,
    ],
    dtype=np.float32,
)

#: The deploy client warns past this token magnitude (run_vla_inference.py:291).
TOKEN_WARN_ABS = 1.25

__all__ = [
    "CONTROL_HZ",
    "HARDWARE_TO_ISAACLAB",
    "HISTORY_LEN",
    "ISAACLAB_TO_HARDWARE",
    "NUM_JOINTS",
    "OBS_DIM",
    "OBS_LAYOUT",
    "SONIC_ACTION_SCALE",
    "SONIC_BASE_HEIGHT",
    "SONIC_DEFAULT_ANGLES",
    "SONIC_JOINT_NAMES",
    "SONIC_KDS",
    "SONIC_KPS",
    "STANDING_TOKEN",
    "TOKEN_DIM",
    "TOKEN_WARN_ABS",
]
