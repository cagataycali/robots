"""Scripted expert and labeller for the S1V typed-decision data.

The expert reads privileged state (:class:`~strands_robots.policies.s1v.scene.Privileged`)
and answers the same typed questions the learned model will: *which* joint to
step, in which *direction*, at which *size*, plus the ``noul`` gates. It is a
greedy one-step planner in primitive space: for each of the 30 arm primitives
it linearises the tool-centre-point motion through the MuJoCo Jacobian and
picks the one whose predicted position-plus-orientation cost is lowest; the
gripper primitives are chosen by a small phase machine (open before the
approach, close at the grasp pose, hold while lifting).

Two tasks share the machine:

* ``reach``: bring the TCP to 2 cm above the cube centre.
* ``pick``: open, approach, descend, close on the cube, lift it 5 cm.

The labeller turns a pair of privileged snapshots (before / after executing a
primitive) into the ``noul`` targets ``cube_in_jaws``, ``progress_if_executed``
and ``safe``. Everything here is torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from strands_robots.policies.s1v.primitives import (
    DEFAULT_GRIPPER_STEP_PCT,
    DEFAULT_STEP_DEG,
    GRIPPER_JOINT,
    HOLD,
    SIZE_LABELS,
    SO101_ARM_LABELS,
    SO101_CTRL_BOUNDS,
    SO101_GRIPPER_RANGE,
    SO101_LABELS,
    Primitive,
    apply_primitive,
    gripper_pct_from_rad,
)
from strands_robots.policies.s1v.scene import CUBE_HALF_Z, Privileged, So101Scene

TASKS: tuple[str, ...] = ("reach", "pick")
PHASES: tuple[str, ...] = ("open", "approach", "descend", "grasp", "lift", "done")
DOWN = np.array([0.0, 0.0, -1.0])
PRE_GRASP_Z = 0.06
#: Grasp height above the cube centre. Keeps the fingertip ~4 mm off the table at
#: a 30 degree jaw tilt: a fingertip on the table pins the whole arm (the so101
#: position servos cannot drag it) and the setpoint winds up.
GRASP_Z = 0.006
#: Anti-windup: an arm setpoint may lead the measured joint by at most this much.
WINDUP_MAX = math.radians(8.0)
REACH_Z = 0.02
#: Reach flies to the goal at this extra height until it is laterally within FLY_XY_TOL.
FLY_Z = 0.04
FLY_XY_TOL = 0.02
LIFT_Z = 0.10
SUCCESS_LIFT = 0.05
REACH_TOL = 0.02
GRASP_TOL = 0.012
GRASP_TOL_X = 0.008
GRASP_TOL_Y = 0.005
GRASP_TOL_Z = 0.012
OPEN_PCT = 60.0
#: Pad-to-pad opening that means the jaws are squeezing the 2 cm cube (metres).
GRASPED_GAP = 0.04
ROT_WEIGHT = 0.002
LEVEL_WEIGHT = 0.002
#: Weight of the jaw-orientation rows in the IK step (1 rad of jaw error ~ 2 cm of TCP error).
ROT_TASK_WEIGHT = 0.05
IK_DAMPING = 1e-3
#: A primitive may not take the TCP more than this below the current goal height.
FLOOR_SLACK = 0.005
#: Jaw tilt from vertical the pick expert tolerates before the orientation term kicks in.
TILT_TOL_COS = math.cos(math.radians(20.0))
#: |z component| of the closing axis the pick expert accepts as "level" before descending.
LEVEL_TOL = 0.25
MIN_GAIN = 1e-5
#: A primitive counts as progress when the phase cost drops by more than this (metres).
PROGRESS_EPS = 5e-4
#: The expert refuses arm setpoints closer than this to a joint limit (radians).
LIMIT_MARGIN = math.radians(1.0)
ARM_PRIMITIVES: tuple[Primitive, ...] = tuple(
    Primitive(j, s, size) for j in SO101_ARM_LABELS for s in (1.0, -1.0) for size in DEFAULT_STEP_DEG
)


@dataclass
class ExpertState:
    """Phase-machine memory carried across ticks of one episode."""

    task: str
    phase: str = "open"
    cube_start: np.ndarray | None = None


def _rest_dict(qpos: np.ndarray) -> dict[str, float]:
    return {str(i + 1): float(q) for i, q in enumerate(qpos)}


def target_for(priv: Privileged, st: ExpertState) -> np.ndarray:
    """Where the TCP should go in the current phase."""
    cube = priv.cube
    if st.task == "reach":
        goal = cube + np.array([0.0, 0.0, REACH_Z])
        if np.linalg.norm((goal - priv.tcp)[:2]) > FLY_XY_TOL:  # fly in at height, then drop onto the goal
            goal = goal + np.array([0.0, 0.0, FLY_Z])
        return goal
    if st.phase in ("open", "approach"):
        return cube + np.array([0.0, 0.0, PRE_GRASP_Z])
    if st.phase in ("descend", "grasp"):
        return cube + np.array([0.0, 0.0, GRASP_Z])
    start = st.cube_start if st.cube_start is not None else cube
    return np.array([start[0], start[1], CUBE_HALF_Z + LIFT_Z])


def position_weights(st: ExpertState) -> np.ndarray:
    """Per-axis weights on the TCP error; lateral error dominates above the cube."""
    return np.array([4.0, 4.0, 1.0]) if (st.task == "pick" and st.phase == "approach") else np.ones(3)


def pose_cost(err: np.ndarray, jaw_axis: np.ndarray, close_axis: np.ndarray, st: ExpertState) -> float:
    """The scalar the pick/reach expert descends: weighted TCP error plus two jaw terms.

    * tilt: the jaws may lean up to 30 degrees from vertical for free; beyond
      that the fingertip reaches the table before the TCP reaches the cube.
    * level: the closing axis of the jaw should be horizontal, so the cube is
      pinched from the sides rather than from above.
    """
    c = float(np.sum(position_weights(st) * err**2))
    if st.task == "pick":
        tilt_excess = max(0.0, TILT_TOL_COS - float(-jaw_axis[2]))
        c += ROT_WEIGHT * tilt_excess**2 + LEVEL_WEIGHT * float(close_axis[2]) ** 2
    return c


def phase_cost(priv: Privileged, st: ExpertState) -> float:
    """Scalar the ``progress_if_executed`` label is measured on."""
    if st.phase == "grasp":
        return priv.gap + (0.0 if priv.cube_in_jaws else 0.05)
    if st.phase == "lift":
        return -float(priv.cube[2]) + (0.0 if priv.cube_in_jaws else 0.5)
    return math.sqrt(pose_cost(target_for(priv, st) - priv.tcp, priv.jaw_axis, priv.close_axis, st))


def advance_phase(priv: Privileged, st: ExpertState) -> None:
    """Move the phase machine forward given the current snapshot."""
    pct = gripper_pct_from_rad(priv.qpos[5], SO101_GRIPPER_RANGE)
    if st.cube_start is None:
        st.cube_start = priv.cube.copy()
    if st.task == "reach":
        st.phase = "done" if np.linalg.norm(target_for(priv, st) - priv.tcp) < REACH_TOL else "approach"
        return
    if st.phase == "open" and pct >= OPEN_PCT - 1.0:
        st.phase = "approach"
    if st.phase == "approach":
        err = target_for(priv, st) - priv.tcp
        oriented = float(-priv.jaw_axis[2]) >= TILT_TOL_COS and abs(float(priv.close_axis[2])) < LEVEL_TOL
        if np.linalg.norm(err[:2]) < 0.015 and abs(err[2]) < 0.025 and oriented:
            st.phase = "descend"  # only descend with the jaws vertical and level: tilted pads topple the block
    if st.phase == "descend":
        if grasp_aligned(priv, st):
            st.phase = "grasp"
    if st.phase == "grasp":
        if priv.cube_in_jaws and priv.gap < GRASPED_GAP:
            st.phase = "lift"
        elif pct <= 0.5 and not priv.cube_in_jaws:
            st.phase = "open"  # missed: reopen and retry from the approach
    if st.phase == "lift" and priv.cube_in_jaws and priv.cube[2] > (st.cube_start[2] + SUCCESS_LIFT):
        st.phase = "done"


def grasp_aligned(priv: Privileged, st: ExpertState) -> bool:
    """Is the cube inside the jaws' sweep? Checked per axis in the static pad's frame.

    The pads are 3 mm thin across (pad y), so a cube 1 cm off sideways is
    missed even though the Euclidean error looks small; along the closing
    axis (pad x) the moving jaw forgives more, and along the jaws (pad z) the
    4 cm pads forgive the most.
    """
    err = priv.pad_rot.T @ (target_for(priv, st) - priv.tcp)
    return bool(abs(err[0]) < GRASP_TOL_X and abs(err[1]) < GRASP_TOL_Y and abs(err[2]) < GRASP_TOL_Z)


def is_success(priv: Privileged, st: ExpertState) -> bool:
    if st.task == "reach":
        return bool(np.linalg.norm(target_for(priv, st) - priv.tcp) < REACH_TOL)
    start = st.cube_start if st.cube_start is not None else priv.cube
    return bool(priv.cube_in_jaws and priv.cube[2] > start[2] + SUCCESS_LIFT)


def expert_primitive(
    scene: So101Scene, priv: Privileged, st: ExpertState, setpoint: dict[str, float] | None = None
) -> Primitive:
    """The expert's answer for the current tick (after :func:`advance_phase`).

    Args:
        scene: The rig (for the Jacobian).
        priv: Current privileged snapshot.
        st: Phase machine state.
        setpoint: The joint targets currently commanded. The so101 position
            servos sag under gravity (kp 17.8, ~5 degree steady-state error), so
            primitives step the *setpoint*, not the measured pose; ``None``
            uses the measured pose.
    """
    pct = gripper_pct_from_rad(priv.qpos[5], SO101_GRIPPER_RANGE)
    if st.phase == "done":
        return HOLD
    if st.task == "pick":
        if st.phase == "open":
            return Primitive(GRIPPER_JOINT, 1.0, "large")
        if st.phase == "grasp":
            return Primitive(GRIPPER_JOINT, -1.0, "large")
        if st.phase in ("approach", "descend") and pct < OPEN_PCT - 1.0:
            return Primitive(GRIPPER_JOINT, 1.0, "large")
    return _best_arm_primitive(scene, priv, st, setpoint)


def _best_arm_primitive(
    scene: So101Scene, priv: Privileged, st: ExpertState, setpoint: dict[str, float] | None
) -> Primitive:
    """Damped least-squares IK step, quantised to the single-joint vocabulary.

    A greedy per-primitive search on the predicted cost cycles here: the
    forward-and-down direction the descent needs is not spanned by any one
    joint, so shoulder_lift (back+down) and elbow_flex (forward+up) alternate
    with no net motion. Instead the expert solves one damped least-squares
    step for the coordinated joint motion that reduces the TCP error and the
    two jaw terms, then executes its largest component as a primitive
    (Gauss-Southwell coordinate descent on the IK objective, recomputed every
    tick, so it converges instead of oscillating).
    """
    jp, jr = scene.tcp_jacobian()
    target = target_for(priv, st)
    err = target - priv.tcp
    weights = position_weights(st)
    rows = [np.sqrt(weights)[:, None] * jp]
    rhs = [np.sqrt(weights) * err]
    if st.task == "pick":
        omega = np.zeros(3)
        jaw, close = priv.jaw_axis, priv.close_axis
        if float(-jaw[2]) < TILT_TOL_COS:  # outside the 30 degree cone: rotate the jaws toward vertical
            omega += np.cross(jaw, DOWN)
        level = np.array([close[0], close[1], 0.0])
        if np.linalg.norm(level) > 1e-6:
            omega += np.cross(close, level / np.linalg.norm(level))
        rows.append(ROT_TASK_WEIGHT * jr)
        rhs.append(ROT_TASK_WEIGHT * omega)
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    dq = np.linalg.solve(a.T @ a + IK_DAMPING * np.eye(5), a.T @ b)
    predicted = float(np.linalg.norm(jp @ dq))
    if predicted > float(np.linalg.norm(err)) > 0.0:  # never plan past the target in one tick
        dq *= float(np.linalg.norm(err)) / predicted
    current = _rest_dict(priv.qpos) if setpoint is None else dict(setpoint)
    for col in np.argsort(-np.abs(dq)):
        col = int(col)
        if abs(dq[col]) < math.radians(0.5):
            break
        key = str(col + 1)
        sign = 1.0 if dq[col] > 0 else -1.0
        wanted = math.degrees(abs(dq[col]))
        fitting = [n for n, deg in DEFAULT_STEP_DEG.items() if deg <= 1.6 * wanted] or [SIZE_LABELS[0]]
        size = min(fitting, key=lambda name: abs(DEFAULT_STEP_DEG[name] - wanted))
        prim = Primitive(SO101_ARM_LABELS[col], sign, size)
        nxt = apply_primitive(
            prim, current, SO101_LABELS, gripper_range=SO101_GRIPPER_RANGE, ctrl_bounds=SO101_CTRL_BOUNDS
        )
        lo, hi = SO101_CTRL_BOUNDS[key]
        if abs(nxt[key] - current[key]) < 1e-6 or nxt[key] <= lo + LIMIT_MARGIN or nxt[key] >= hi - LIMIT_MARGIN:
            continue  # clipped at, or leaning on, the joint limit: unsafe or no motion
        dz = float(jp[2, col] * (nxt[key] - current[key]))
        if priv.tcp[2] + dz < target[2] - FLOOR_SLACK and dz < 0.0:
            continue  # would dive below the current goal height: that is how fingertips meet the table
        return prim
    return HOLD


def labels_after(before: Privileged, after: Privileged, st_before_phase: str, st: ExpertState) -> dict[str, float]:
    """``noul`` targets for the tick whose primitive turned ``before`` into ``after``."""
    probe = ExpertState(task=st.task, phase=st_before_phase, cube_start=st.cube_start)
    progressed = phase_cost(after, probe) < phase_cost(before, probe) - PROGRESS_EPS
    return {
        "cube_in_jaws": float(after.cube_in_jaws),
        "progress_if_executed": float(progressed),
        "safe": float(not (after.limit_hit or after.table_hit)),
    }


def run_episode(
    scene: So101Scene,
    task: str,
    *,
    actor: Callable[[dict[str, Any], Privileged, Primitive], Primitive] | None = None,
    max_ticks: int = 120,
    hz: float = 10.0,
    randomize: bool = True,
    keep_images: bool = True,
    on_tick: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Roll one episode; the expert labels every tick, ``actor`` decides what executes.

    Args:
        scene: The rig.
        task: ``"reach"`` or ``"pick"``.
        actor: ``(observation, privileged, expert_primitive) -> primitive`` to
            execute. ``None`` executes the expert (round-0 data). A learner
            actor gives DAgger data: the learner drives, the expert labels.
        max_ticks: Episode length cap.
        hz: Control rate; primitives are applied once per tick.
        randomize: Cube pose / light / camera jitter on reset.
        keep_images: Keep the two camera arrays on each tick record.
        on_tick: Optional sink called with each tick record (streaming writers).

    Returns:
        ``{"task", "success", "ticks", "steps_to_success", "final_distance",
        "episode": episode_params, "records": [tick records]}``. A tick record
        carries ``state`` (6 floats: deg x5 + gripper %), ``expert`` (the label
        primitive), ``executed``, ``labels`` (noul dict), ``phase`` and, when
        kept, ``scene`` / ``wrist`` uint8 images.
    """
    if task not in TASKS:
        raise ValueError(f"run_episode: task must be one of {TASKS}, got {task!r}")
    params = scene.reset_episode(randomize=randomize)
    st = ExpertState(task=task)
    records: list[dict[str, Any]] = []
    steps_to_success: int | None = None
    priv = scene.snapshot()
    setpoint = _rest_dict(priv.qpos)
    for tick in range(max_ticks):
        advance_phase(priv, st)
        obs = scene.observation()
        expert = expert_primitive(scene, priv, st, setpoint)
        executed = expert if actor is None else actor(obs, priv, expert)
        phase = st.phase
        action = apply_primitive(
            executed, setpoint, SO101_LABELS, gripper_range=SO101_GRIPPER_RANGE, ctrl_bounds=SO101_CTRL_BOUNDS
        )
        action = clamp_windup(action, priv.qpos)
        setpoint = action
        scene.apply(action, hz=hz)
        after = scene.snapshot()
        rec: dict[str, Any] = {
            "tick": tick,
            "phase": phase,
            "state": state_vector(priv.qpos),
            "expert": expert.as_dict(),
            "executed": executed.as_dict(),
            "labels": labels_after(priv, after, phase, st),
            "setpoint_qpos": [action[str(i)] for i in range(1, 7)],
            "tcp_cube_distance": float(np.linalg.norm(priv.cube - priv.tcp)),
        }
        if keep_images:
            rec["scene"] = obs["scene"]
            rec["wrist"] = obs["wrist"]
        records.append(rec)
        if on_tick is not None:
            on_tick(rec)
        priv = after
        if steps_to_success is None and is_success(priv, st):
            steps_to_success = tick + 1
            advance_phase(priv, st)
            if actor is None:
                break
    final = scene.snapshot()
    return {
        "task": task,
        "success": is_success(final, st),
        "ticks": len(records),
        "steps_to_success": steps_to_success,
        "final_distance": float(
            np.linalg.norm(target_for(final, ExpertState(task, "approach", st.cube_start)) - final.tcp)
        )
        if task == "reach"
        else float(np.linalg.norm(final.cube - final.tcp)),
        "cube_lift": float(final.cube[2] - (st.cube_start[2] if st.cube_start is not None else CUBE_HALF_Z)),
        "episode": params,
        "records": records,
    }


def clamp_windup(action: dict[str, float], qpos: np.ndarray, limit: float = WINDUP_MAX) -> dict[str, float]:
    """Keep every arm setpoint within ``limit`` of the measured joint (gripper untouched)."""
    out = dict(action)
    for i in range(5):
        key = str(i + 1)
        out[key] = float(min(max(action[key], qpos[i] - limit), qpos[i] + limit))
    return out


def state_vector(qpos: np.ndarray) -> list[float]:
    """Proprio the model sees: five arm joints in degrees + gripper opening percent."""
    return [math.degrees(float(q)) for q in qpos[:5]] + [gripper_pct_from_rad(float(qpos[5]), SO101_GRIPPER_RANGE)]
