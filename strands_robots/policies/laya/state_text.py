"""Serialize a robot observation into the text state Laya reads, and build its questions.

Laya is text-only: it has no image encoder and no continuous output head, so
the only thing it can ever see of a robot is what this module writes down. The
state is a JSON-able dict (Laya accepts ``dict`` states directly) holding the
instruction, joint angles in degrees, the gripper opening in percent and, when
a privileged world reader is installed, the cube and gripper positions, their
difference vector, the distance and the contact flags. Nothing in here is
estimated: every field is either read from the observation or from the
simulator, or absent.

Questions follow the Laya typed-decision schema: ``choice`` (labelled
options), ``score`` (ordered levels) and ``noul`` (a probability that a
statement holds).
"""

from __future__ import annotations

import math
from typing import Any

from .primitives import GRIPPER_JOINT, NONE_JOINT, SIZE_LABELS

#: so101 registry joint labels, keyed by sim state key. The hardware keys are
#: ``"<label>.pos"``; :func:`labels_for_keys` handles both shapes.
SO101_JOINT_LABELS: dict[str, str] = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}

#: Question profiles this provider knows. Each is a tuple of question names;
#: :func:`build_questions` writes the actual prompts.
QUESTION_PROFILES: dict[str, tuple[str, ...]] = {
    # The smallest set that yields a primitive: 3 questions, ~1 forward pass each.
    "joint_direction_size": ("joint", "direction", "size"),
    # Adds the two calibration gates used by H2 (progress / grasp).
    "joint_direction_size_gated": ("joint", "direction", "size", "progress_ok", "cube_in_jaws"),
}


def labels_for_keys(robot_state_keys: list[str], joint_labels: dict[str, str] | None = None) -> dict[str, str]:
    """Name each robot state key in the question vocabulary.

    Args:
        robot_state_keys: The keys the runtime bound via ``set_robot_state_keys``
            (sim so101: ``"1".."6"``; hardware: ``"shoulder_pan.pos"`` ...).
        joint_labels: Explicit ``{state_key: label}`` override. When ``None``
            the so101 sim keys map through :data:`SO101_JOINT_LABELS`, a
            ``<label>.pos`` key maps to ``<label>``, and any other key names
            itself.

    Returns:
        ``{state_key: label}`` for every key, labels distinct.

    Raises:
        ValueError: If two keys resolve to the same label - the ``joint``
            question could not tell them apart.
    """
    labels: dict[str, str] = {}
    for key in robot_state_keys:
        if joint_labels is not None and key in joint_labels:
            labels[key] = str(joint_labels[key])
        elif key in SO101_JOINT_LABELS:
            labels[key] = SO101_JOINT_LABELS[key]
        elif key.endswith(".pos"):
            labels[key] = key[: -len(".pos")]
        else:
            labels[key] = key
    if len(set(labels.values())) != len(labels):
        raise ValueError(f"labels_for_keys: labels are not distinct: {labels}")
    return labels


def _round_list(values: Any, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def serialize_state(
    *,
    instruction: str,
    joints_rad: dict[str, float],
    labels: dict[str, str],
    gripper_pct: float | None,
    world: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON state Laya reads for one tick.

    Args:
        instruction: The task text (``reads_instruction`` is ``True``).
        joints_rad: Current joint values keyed by state key, radians.
        labels: ``{state_key: label}`` from :func:`labels_for_keys`.
        gripper_pct: Gripper opening in percent, or ``None`` when the robot has
            no labelled gripper.
        world: Privileged sim readings, when a world reader is installed:
            ``gripper_xyz_m``, ``cube_xyz_m`` (lists of 3 floats) and
            ``contacts`` (list of strings). Anything else is copied through
            if JSON-native.

    Returns:
        The state dict. Angles are rounded to 0.1 deg and positions to 1 mm so
        the same physical state serializes to the same text.
    """
    joints_deg = {
        label: round(math.degrees(float(joints_rad[key])), 1)
        for key, label in labels.items()
        if label != GRIPPER_JOINT and key in joints_rad
    }
    state: dict[str, Any] = {"instruction": instruction, "joints_deg": joints_deg}
    if gripper_pct is not None:
        state["gripper_open_pct"] = round(float(gripper_pct), 1)
    if world is None:
        state["world"] = "not observed (no privileged reader installed; joints only)"
        return state
    gripper = world.get("gripper_xyz_m")
    cube = world.get("cube_xyz_m")
    if gripper is not None:
        state["gripper_xyz_m"] = _round_list(gripper, 3)
    if cube is not None:
        state["cube_xyz_m"] = _round_list(cube, 3)
    if gripper is not None and cube is not None:
        delta = [float(c) - float(g) for g, c in zip(gripper, cube, strict=True)]
        state["gripper_to_cube_m"] = _round_list(delta, 3)
        state["distance_to_cube_m"] = round(math.sqrt(sum(d * d for d in delta)), 3)
    if "contacts" in world:
        state["contacts"] = [str(c) for c in world["contacts"]]
    for key, value in world.items():
        if key not in state and key not in ("gripper_xyz_m", "cube_xyz_m", "contacts"):
            if isinstance(value, (str, int, float, bool)) or value is None:
                state[key] = value
    return state


def build_questions(arm_labels: tuple[str, ...], profile: str = "joint_direction_size") -> dict[str, Any]:
    """The Laya questions for one control tick.

    Args:
        arm_labels: Arm joint labels in the ``joint`` choice, in order.
        profile: One of :data:`QUESTION_PROFILES`.

    Returns:
        ``{question_name: question}`` in Laya's typed-decision schema.

    Raises:
        ValueError: If ``profile`` is unknown or ``arm_labels`` is empty.
    """
    if profile not in QUESTION_PROFILES:
        raise ValueError(f"build_questions: unknown profile {profile!r}; expected one of {sorted(QUESTION_PROFILES)}")
    if not arm_labels:
        raise ValueError("build_questions: arm_labels must not be empty")
    criteria = {label: f"rotate the {label.replace('_', ' ')} joint" for label in arm_labels}
    criteria[GRIPPER_JOINT] = "open or close the gripper"
    criteria[NONE_JOINT] = "hold every joint where it is"
    library: dict[str, Any] = {
        "joint": {
            "type": "choice",
            "instructions": (
                "The state describes a robot arm, its gripper and a cube on the table, all in metres and degrees. "
                "Which single joint should move next so the gripper gets closer to the cube and the task progresses?"
            ),
            "criteria": criteria,
        },
        "direction": {
            "type": "choice",
            "instructions": "Should that joint's angle increase or decrease?",
            "criteria": {
                "positive": "increase the joint angle (for the gripper: open it)",
                "negative": "decrease the joint angle (for the gripper: close it)",
            },
        },
        "size": {
            "type": "score",
            "instructions": "How large should this step be, given the remaining distance to the cube?",
            "criteria": list(SIZE_LABELS),
        },
        "progress_ok": {
            "type": "noul",
            "instructions": "Will the chosen step reduce the distance between the gripper and the cube?",
        },
        "cube_in_jaws": {
            "type": "noul",
            "instructions": "Is the cube currently between the gripper jaws, close enough to grasp?",
        },
    }
    return {name: library[name] for name in QUESTION_PROFILES[profile]}
