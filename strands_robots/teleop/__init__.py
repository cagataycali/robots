"""Whole-body teleoperation: sources, retargeting, encoders and recording layouts.

See ``docs/project/design-wholebody-teleop.md`` for the design this package
implements piece by piece. What ships here is what the design proves without
hardware or model weights: the frame type, a scripted mock source, a joint-map
retarget, the composing :class:`WholeBodyTeleoperator` and the recording
layouts a G1 dataset can be written in.
"""

from strands_robots.teleop.layouts import (
    G1_HARDWARE_JOINTS,
    G1_LEROBOT_JOINTS,
    G1_SIM_JOINTS,
    GRIPPERS,
    ISAACLAB_TO_HARDWARE,
    TOKEN_DIM,
    Layout,
    list_layouts,
    teleop_layout,
)
from strands_robots.teleop.wholebody import (
    Encoder,
    JointMapRetarget,
    MockPoseSource,
    PoseFrame,
    PoseSource,
    Retarget,
    RetargetOut,
    WholeBodyTeleoperator,
)

__all__ = [
    "G1_HARDWARE_JOINTS",
    "G1_LEROBOT_JOINTS",
    "G1_SIM_JOINTS",
    "GRIPPERS",
    "ISAACLAB_TO_HARDWARE",
    "TOKEN_DIM",
    "Encoder",
    "JointMapRetarget",
    "Layout",
    "MockPoseSource",
    "PoseFrame",
    "PoseSource",
    "Retarget",
    "RetargetOut",
    "WholeBodyTeleoperator",
    "list_layouts",
    "teleop_layout",
]
