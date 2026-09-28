"""Discrete action primitives for a typed-decision (System 1) policy.

Laya answers typed questions; it never emits a number for a joint. So the
control surface of :class:`~strands_robots.policies.laya.policy.LayaPolicy` is
a small set of *primitives*: move ONE joint (or the gripper) ONE step in ONE
direction, at one of three step sizes, or hold. This module owns that
vocabulary and the arithmetic that turns a primitive plus the current joint
state into a joint-target action dict. Everything here is pure Python and
torch-free so it can be unit-tested without a model.

Sign conventions: an arm joint step is ``+/- step_deg[size]`` converted to
radians and added to the joint's current position, clipped to the actuator's
ctrl range when one is known. A gripper step is ``+/- gripper_step_pct``
percent of the gripper's ctrl range (``0`` = closed = range low, ``100`` =
open = range high, the so101 registry convention), so ``positive`` opens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: The step-size labels, low to high. Laya's ``score`` question type returns a
#: float on the index scale of its criteria list (``0 .. len - 1``), so this
#: order is also the decode order.
SIZE_LABELS: tuple[str, ...] = ("small", "medium", "large")

#: Default arm step per size, in degrees.
DEFAULT_STEP_DEG: dict[str, float] = {"small": 2.0, "medium": 5.0, "large": 10.0}

#: Default gripper step in percent of the gripper's ctrl range.
DEFAULT_GRIPPER_STEP_PCT: float = 15.0

#: Name of the hold primitive (Laya picks it from the joint choice).
NONE_JOINT: str = "none"

#: Name of the gripper joint in the question vocabulary.
GRIPPER_JOINT: str = "gripper"

#: Direction labels and their signs.
DIRECTIONS: dict[str, float] = {"positive": 1.0, "negative": -1.0}


@dataclass(frozen=True)
class Primitive:
    """One discrete control decision.

    Attributes:
        joint: Joint label to move (an arm label, :data:`GRIPPER_JOINT`, or
            :data:`NONE_JOINT` for hold).
        direction: ``+1.0`` or ``-1.0``; ignored when ``joint`` is the hold.
        size: One of :data:`SIZE_LABELS`; ignored when ``joint`` is the hold.
    """

    joint: str
    direction: float = 1.0
    size: str = "medium"

    @property
    def is_hold(self) -> bool:
        """Whether this primitive leaves every joint where it is."""
        return self.joint == NONE_JOINT

    def as_dict(self) -> dict[str, Any]:
        """JSON-native form for tick logs."""
        return {"joint": self.joint, "direction": self.direction, "size": self.size}


HOLD = Primitive(NONE_JOINT, 0.0, "small")


def size_from_score(score: float, labels: tuple[str, ...] = SIZE_LABELS) -> str:
    """Map a Laya ``score`` (index-scale float) onto a size label.

    Args:
        score: The float Laya returned; ``0`` means the first label and
            ``len(labels) - 1`` the last. Values outside are clipped.
        labels: The criteria list the question was asked with.

    Returns:
        The nearest label.

    Raises:
        ValueError: If ``score`` is not finite or ``labels`` is empty.
    """
    if not labels:
        raise ValueError("size_from_score: labels must not be empty")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError(f"size_from_score: score must be a finite number, got {score!r}")
    index = int(round(min(max(float(score), 0.0), float(len(labels) - 1))))
    return labels[index]


def gripper_pct_from_rad(value_rad: float, gripper_range: tuple[float, float]) -> float:
    """Gripper opening in percent (0 closed .. 100 open) from an actuator value."""
    low, high = gripper_range
    if high == low:
        raise ValueError("gripper_pct_from_rad: gripper_range must span a non-zero interval")
    return 100.0 * (float(value_rad) - low) / (high - low)


def gripper_rad_from_pct(pct: float, gripper_range: tuple[float, float]) -> float:
    """Inverse of :func:`gripper_pct_from_rad`, clipped to the range."""
    low, high = gripper_range
    pct = min(max(float(pct), 0.0), 100.0)
    return low + (high - low) * pct / 100.0


def apply_primitive(
    primitive: Primitive,
    current: dict[str, float],
    labels: dict[str, str],
    *,
    step_deg: dict[str, float] | None = None,
    gripper_step_pct: float = DEFAULT_GRIPPER_STEP_PCT,
    gripper_range: tuple[float, float],
    ctrl_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Turn a primitive into the next joint-target action dict.

    Args:
        primitive: The decision to apply.
        current: Current joint positions keyed by robot state key (radians;
            the gripper value is its actuator value in the same units).
        labels: ``{state_key: label}`` naming each state key in the question
            vocabulary (so101: ``{"1": "shoulder_pan", ..., "6": "gripper"}``).
        step_deg: Arm step per size label in degrees; :data:`DEFAULT_STEP_DEG`
            when ``None``.
        gripper_step_pct: Gripper step in percent of its range.
        gripper_range: ``(closed, open)`` actuator values of the gripper.
        ctrl_bounds: Optional ``{state_key: (low, high)}`` to clip arm targets.

    Returns:
        ``{state_key: target}`` for EVERY key in ``current`` (unchanged joints
        keep their current value so the runner sees a complete action), with
        python floats only.

    Raises:
        ValueError: If the primitive names a label that no state key carries,
            or a size that :data:`SIZE_LABELS` does not.
    """
    steps = DEFAULT_STEP_DEG if step_deg is None else step_deg
    action = {key: float(value) for key, value in current.items()}
    if primitive.is_hold:
        return action
    if primitive.size not in steps:
        raise ValueError(f"apply_primitive: unknown size {primitive.size!r}; expected one of {sorted(steps)}")
    by_label = {label: key for key, label in labels.items()}
    key = by_label.get(primitive.joint)
    if key is None or key not in action:
        raise ValueError(
            f"apply_primitive: primitive joint {primitive.joint!r} is not one of the labelled state keys "
            f"{sorted(by_label)}"
        )
    sign = 1.0 if primitive.direction >= 0 else -1.0
    if primitive.joint == GRIPPER_JOINT:
        pct = gripper_pct_from_rad(action[key], gripper_range) + sign * float(gripper_step_pct)
        action[key] = gripper_rad_from_pct(pct, gripper_range)
        return action
    target = action[key] + sign * math.radians(float(steps[primitive.size]))
    bound = (ctrl_bounds or {}).get(key)
    if bound is not None:
        target = min(max(target, bound[0]), bound[1])
    action[key] = float(target)
    return action


def decode_answers(answers: dict[str, Any], arm_labels: tuple[str, ...]) -> tuple[Primitive, dict[str, float]]:
    """Read a primitive and its confidences out of Laya's ``answers``.

    Expects the questions built by
    :func:`~strands_robots.policies.laya.state_text.build_questions`: ``joint``
    (choice over ``arm_labels + (gripper, none)``), ``direction`` (choice
    positive/negative), ``size`` (score over :data:`SIZE_LABELS`) and the
    optional ``noul`` gates ``progress_ok`` and ``cube_in_jaws``.

    Args:
        answers: ``result["answers"]`` from ``Router.predict``.
        arm_labels: The arm joint labels the ``joint`` question offered.

    Returns:
        ``(primitive, confidences)`` where ``confidences`` carries
        ``joint`` / ``direction`` / ``size`` (Laya's ``confidence`` field),
        ``joint_p`` (probability of the chosen joint) and, when asked,
        ``progress_ok`` / ``cube_in_jaws`` (the ``noul`` probabilities).

    Raises:
        KeyError: If a required question is missing from ``answers``.
        ValueError: If the chosen joint is not in the offered vocabulary.
    """
    joint = answers["joint"]
    direction = answers["direction"]
    size = answers["size"]
    chosen = str(joint["choice"])
    allowed = set(arm_labels) | {GRIPPER_JOINT, NONE_JOINT}
    if chosen not in allowed:
        raise ValueError(f"decode_answers: Laya chose joint {chosen!r}, not one of {sorted(allowed)}")
    sign = DIRECTIONS.get(str(direction["choice"]))
    if sign is None:
        raise ValueError(f"decode_answers: Laya chose direction {direction['choice']!r}, expected positive/negative")
    primitive = Primitive(chosen, sign, size_from_score(float(size["score"])))
    confidences: dict[str, float] = {
        "joint": float(joint.get("confidence", 0.0)),
        "joint_p": float(joint.get("probabilities", {}).get(chosen, joint.get("answer_confidence", 0.0))),
        "direction": float(direction.get("confidence", 0.0)),
        "size": float(size.get("confidence", 0.0)),
    }
    for gate in ("progress_ok", "cube_in_jaws"):
        if gate in answers and "noul" in answers[gate]:
            confidences[gate] = float(answers[gate]["noul"])
    return primitive, confidences
