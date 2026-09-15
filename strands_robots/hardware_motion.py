"""Direct joint motion on a real arm: small, gated, read back.

The real-hardware tool's only motion verbs were ``execute`` and ``start`` - a
policy rollout. An agent asked to "move the wrist 5 degrees" had no action for
it: the tool description named no ``set_joint_positions``, so the agent either
guessed at a rollout instruction or reported the arm could not be moved. This
module is the direct path, shaped by three facts about real arms:

* **Every write is gated and bounded.** The caller runs the operator gate
  before anything here is reached; this module then refuses any joint asked
  to travel more than a per-call cap (``config.max_relative_target`` when the
  arm declares one, else :data:`DEFAULT_STEP_CAP_DEG`) - refuses, rather than
  clamping the way lerobot's ``send_action`` does, because a clamped motion
  is a motion the operator did not approve.
* **The arm may be uncalibrated.** lerobot's normalised write needs the
  calibration file; without it the joint is commanded in the encoder frame
  :func:`~strands_robots.hardware_observe.read_joint_state` reports
  (``2048 ticks = 0°``), so what ``get_state`` shows and what
  ``set_joint_positions`` accepts are one frame either way. The text says
  which frame was used.
* **Position is what was measured, not what was sent.** After the write the
  bus is read again and each joint reports its target, where it arrived, and
  the error. A servo that stalled on an obstacle is then visible as an error,
  not hidden behind an echoed command.

Torque is enabled on the commanded joints (a servo cannot hold a goal
otherwise) and LEFT ON, which the text states; ``set_torque enabled=false``
releases the arm and is never gated - stopping is never harder than moving.
Duck-typed on lerobot's ``MotorsBus`` like :mod:`~strands_robots.hardware_observe`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from strands_robots.bus_access import bus_lock
from strands_robots.hardware_observe import (
    TICKS_CENTRE,
    TICKS_PER_REV,
    ensure_bus_open,
    read_joint_state,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DIRECT_MOTION_ACTIONS",
    "DEFAULT_STEP_CAP_DEG",
    "REACHED_TOLERANCE_DEG",
    "degrees_to_ticks",
    "step_cap_for",
    "plan_joint_targets",
    "move_joints",
    "set_torque",
    "format_move",
    "format_torque",
]

#: The actions this module answers that MOVE the arm; every one is gated by the caller.
DIRECT_MOTION_ACTIONS: frozenset[str] = frozenset({"set_joint_positions", "set_gripper"})

#: Largest travel one call may ask of one joint when the arm declares no
#: ``max_relative_target``. In the unit the joint reports in (degrees, or
#: percent for a 0-100 gripper).
DEFAULT_STEP_CAP_DEG = 20.0

#: A joint within this many units of its target is reported ``reached``.
REACHED_TOLERANCE_DEG = 2.0

#: How long the servos are given before the read-back, and its ceiling.
DEFAULT_SETTLE_S = 0.5
MAX_SETTLE_S = 3.0

_GOAL_REGISTER = "Goal_Position"
_OPERATING_MODE_REGISTER = "Operating_Mode"
_POSITION_MODE = 0


def degrees_to_ticks(degrees: float) -> int:
    """Inverse of :func:`~strands_robots.hardware_observe.ticks_to_degrees`, clamped to the encoder."""
    ticks = round(TICKS_CENTRE + float(degrees) * TICKS_PER_REV / 360.0)
    return max(0, min(TICKS_PER_REV - 1, ticks))


def step_cap_for(robot: Any, joint: str) -> float:
    """The per-call travel cap for ``joint``: the arm's declared limit, else the default."""
    declared = getattr(getattr(robot, "config", None), "max_relative_target", None)
    if isinstance(declared, Mapping):
        value = declared.get(joint)
        return float(value) if value is not None else DEFAULT_STEP_CAP_DEG
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return float(declared)
    return DEFAULT_STEP_CAP_DEG


def _finite(value: Any, *, label: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return f


def plan_joint_targets(
    robot: Any,
    positions: Mapping[str, Any],
    *,
    raw: bool = False,
) -> dict[str, Any]:
    """Read the arm and decide, joint by joint, what one call may write. Writes nothing.

    Returns ``{"calibrated", "frame", "current", "targets", "deltas", "cap"}``
    where ``frame`` is ``"calibration"``, ``"encoder_estimate"`` or ``"ticks"``
    and ``targets`` maps joint name to the value the write will carry (in
    ``frame``'s unit).

    Raises:
        ValueError: An empty request, an unknown joint (the real names are
            listed), a non-finite value, or a joint asked to travel more than
            its cap - the message names the joint, the travel, the cap and the
            remedy (split the move, or declare ``max_relative_target``).
    """
    if not isinstance(positions, Mapping) or not positions:
        raise ValueError("positions must be a non-empty mapping of joint name -> target, e.g. {'wrist_roll': 12.5}")
    state = read_joint_state(robot)
    joints: Mapping[str, Mapping[str, Any]] = state["joints"]
    unknown = [str(name) for name in positions if str(name) not in joints]
    if unknown:
        raise ValueError(f"Unknown joint(s) {unknown}. This arm's joints: {sorted(joints)}")

    calibrated = bool(state["calibrated"])
    if raw:
        frame = "ticks"
    elif calibrated:
        frame = "calibration"
    else:
        frame = "encoder_estimate"

    current: dict[str, float] = {}
    targets: dict[str, float] = {}
    deltas: dict[str, float] = {}
    caps: dict[str, float] = {}
    for name, value in positions.items():
        name = str(name)
        joint = joints[name]
        target = _finite(value, label=f"positions[{name!r}]")
        if raw:
            now = float(joint["ticks"])
            target = float(int(target))
            cap = step_cap_for(robot, name) * TICKS_PER_REV / 360.0
        elif "degrees" in joint:
            now = float(joint["degrees"])
            cap = step_cap_for(robot, name)
        else:
            # A calibrated 0-100 joint (the gripper) is commanded in its own unit.
            now = float(joint["normalized"])
            cap = step_cap_for(robot, name)
        delta = target - now
        if abs(delta) > cap + 1e-9:
            unit = "ticks" if raw else ("%" if "normalized" in joint else "°")
            raise ValueError(
                f"{name}: asked to travel {delta:+.1f}{unit} (from {now:.1f} to {target:.1f}) but one call may move a "
                f"joint at most {cap:.1f}{unit}. Split the move into steps of {cap:.1f}{unit} or less, or declare a "
                "larger max_relative_target on the robot config."
            )
        current[name] = now
        targets[name] = target
        deltas[name] = delta
        caps[name] = cap
    return {
        "calibrated": calibrated,
        "frame": frame,
        "current": current,
        "targets": targets,
        "deltas": deltas,
        "cap": caps,
        "port": state.get("port"),
    }


def _operating_modes(bus: Any, names: Sequence[str]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for name in names:
        try:
            modes[name] = bus.read(_OPERATING_MODE_REGISTER, name, normalize=False)
        except Exception as exc:  # noqa: BLE001 - a bus without the register is not in a wrong mode
            logger.debug("Operating_Mode read of %s failed: %s", name, exc)
    return modes


def move_joints(
    robot: Any,
    positions: Mapping[str, Any],
    *,
    raw: bool = False,
    settle_s: float = DEFAULT_SETTLE_S,
) -> dict[str, Any]:
    """Command the named joints, wait ``settle_s``, read back. The ONE write path.

    The caller has already run the operator gate. Refuses (via
    :func:`plan_joint_targets`) before anything is written; a servo not in
    position mode is refused too, naming ``configure()`` - this module does not
    rewrite servo modes. Enables torque on the commanded joints and leaves it on.

    Returns the plan plus ``{"reached": {name: bool}, "actual": {name: value},
    "error": {name: value}, "settle_s", "torque_left_on": [names]}``.
    """
    settle = max(0.0, min(MAX_SETTLE_S, _finite(settle_s, label="settle_s")))
    plan = plan_joint_targets(robot, positions, raw=raw)
    names = list(plan["targets"])
    bus = robot.bus
    ensure_bus_open(robot)

    with bus_lock(robot):
        modes = _operating_modes(bus, names)
        wrong = {n: m for n, m in modes.items() if m not in (None, _POSITION_MODE)}
        if wrong:
            raise ValueError(
                f"servo(s) not in position mode (Operating_Mode {wrong}); run the robot's configure() - a policy "
                "rollout does - before commanding positions."
            )
        bus.enable_torque(names)
        if plan["frame"] == "calibration":
            bus.sync_write(_GOAL_REGISTER, {n: plan["targets"][n] for n in names})
        else:
            ticks = {n: int(plan["targets"][n]) if raw else degrees_to_ticks(plan["targets"][n]) for n in names}
            bus.sync_write(_GOAL_REGISTER, ticks, normalize=False)
    logger.info("commanded %s on %s (%s frame)", plan["targets"], plan.get("port"), plan["frame"])

    if settle:
        time.sleep(settle)
    after = read_joint_state(robot)["joints"]
    actual: dict[str, float] = {}
    error: dict[str, float] = {}
    reached: dict[str, bool] = {}
    for n in names:
        joint = after[n]
        if raw:
            got = float(joint["ticks"])
            tol = REACHED_TOLERANCE_DEG * TICKS_PER_REV / 360.0
        elif "degrees" in joint:
            got = float(joint["degrees"])
            tol = REACHED_TOLERANCE_DEG
        else:
            got = float(joint["normalized"])
            tol = REACHED_TOLERANCE_DEG
        actual[n] = got
        error[n] = round(got - plan["targets"][n], 2)
        reached[n] = abs(error[n]) <= tol
    plan.update(
        {
            "reached": reached,
            "actual": actual,
            "error": error,
            "settle_s": settle,
            "torque_left_on": names,
        }
    )
    return plan


def set_torque(robot: Any, enabled: bool, joints: Sequence[str] | None = None) -> dict[str, Any]:
    """Enable or release torque on ``joints`` (all when omitted). Disabling is never gated."""
    state = read_joint_state(robot)
    known = list(state["joints"])
    names = [str(j) for j in joints] if joints else known
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown joint(s) {unknown}. This arm's joints: {sorted(known)}")
    bus = robot.bus
    with bus_lock(robot):
        if enabled:
            bus.enable_torque(names)
        else:
            bus.disable_torque(names)
    after = read_joint_state(robot)["joints"]
    return {
        "enabled": bool(enabled),
        "joints": names,
        "torque_enabled": {n: after[n].get("torque_enabled") for n in names},
        "port": state.get("port"),
    }


def _unit(frame: str, joint: str, plan: Mapping[str, Any]) -> str:
    if frame == "ticks":
        return " ticks"
    return "%" if joint == "gripper" and plan.get("calibrated") else "°"


def format_move(tool_name: str, plan: Mapping[str, Any]) -> str:
    """The text an agent reads after a move: per joint target → actual, then torque and frame."""
    frame = plan["frame"]
    n_reached = sum(1 for v in plan["reached"].values() if v)
    n = len(plan["targets"])
    head = f"{tool_name}: moved {n} joint(s), {n_reached}/{n} reached target within {REACHED_TOLERANCE_DEG:g}"
    head += " ticks" if frame == "ticks" else "°"
    head += f" after {plan['settle_s']:g} s."
    lines = [head]
    for j in plan["targets"]:
        u = _unit(frame, j, plan)
        mark = "reached" if plan["reached"][j] else f"NOT reached (error {plan['error'][j]:+.1f}{u})"
        lines.append(
            f"  {j}: {plan['current'][j]:.1f}{u} → target {plan['targets'][j]:.1f}{u}, "
            f"actual {plan['actual'][j]:.1f}{u} - {mark}"
        )
    lines.append(f"Torque is ON on {', '.join(plan['torque_left_on'])} (holding); set_torque enabled=false releases.")
    if frame == "encoder_estimate":
        lines.append(
            "Frame: encoder estimate (arm NOT calibrated; 2048 ticks = 0°). Joint limits are unknown to the tool - "
            "keep steps small and watch the arm."
        )
    elif frame == "ticks":
        lines.append("Frame: raw encoder ticks.")
    return "\n".join(lines)


def format_torque(tool_name: str, result: Mapping[str, Any]) -> str:
    """One line: which joints now hold and which can be moved by hand."""
    state = "ON (holding position)" if result["enabled"] else "OFF (arm can be moved by hand)"
    return f"{tool_name}: torque {state} on {', '.join(result['joints'])}."
