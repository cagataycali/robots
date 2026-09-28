"""Laya System 1 typed-decision policy (text state -> discrete joint primitive).

Laya (``convaiinnovations/laya``) answers typed questions about a text state in
one non-autoregressive forward pass with calibrated probabilities. It has no
vision and no continuous output, so this provider serializes the robot state
to JSON, asks which joint / direction / step size to apply, and emits ONE
joint-target action per tick. Research provider for the System 1 gate study;
see :mod:`strands_robots.policies.laya.policy`.

Quickstart::

    from strands_robots import Robot

    robot = Robot("so101", mode="sim")
    robot.run_policy(policy_provider="laya", policy_config={"model": "english"},
                     instruction="reach the cube", duration=10, control_frequency=10)
"""

from .policy import LAYA_MODELS, SO101_SIM_GRIPPER_RANGE_RAD, LayaPolicy
from .primitives import HOLD, Primitive, apply_primitive, decode_answers, size_from_score
from .state_text import QUESTION_PROFILES, SO101_JOINT_LABELS, build_questions, labels_for_keys, serialize_state

__all__ = [
    "HOLD",
    "LAYA_MODELS",
    "QUESTION_PROFILES",
    "SO101_JOINT_LABELS",
    "SO101_SIM_GRIPPER_RANGE_RAD",
    "LayaPolicy",
    "Primitive",
    "apply_primitive",
    "build_questions",
    "decode_answers",
    "labels_for_keys",
    "serialize_state",
    "size_from_score",
]
