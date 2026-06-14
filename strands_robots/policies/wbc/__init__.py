"""GR00T Whole-Body-Control (decoupled WBC) policy provider.

Lightweight ONNX-only locomotion policy for humanoids (Unitree G1 and similar).
Uses decoupled architecture: RL lower-body (12 leg + 3 waist joints) with
separate upper-body control (IK / VLA / scripted).

The policy auto-switches between Balance and Walk ONNX checkpoints based on
the magnitude of the commanded velocity.

Reference: https://github.com/NVlabs/GR00T-WholeBodyControl
License: Apache 2.0 (code), NVIDIA Open Model License (weights)

Usage::

    from strands_robots.policies import create_policy

    policy = create_policy("wbc", checkpoint="nvidia/GR00T-WholeBodyControl")
    policy.set_robot_state_keys(WBC_JOINT_NAMES)
    actions = policy.get_actions_sync(obs, "", target_velocity=[0.5, 0, 0])
"""

from strands_robots.policies.wbc.wbc_policy import (
    WBC_JOINT_NAMES,
    WBCPolicy,
)

__all__ = ["WBCPolicy", "WBC_JOINT_NAMES"]
