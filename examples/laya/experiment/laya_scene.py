"""Shared scene + privileged world reader + baselines for the Laya System 1 lane (sim only).

so101 in MuJoCo at the F3A rest pose (sticky across reset()), a red 2 cm cube on the table in front of the arm
(forward = -y), scene + wrist cameras for the LeRobot recording. The world reader reads the fingertip site
``so101/gripper`` and the ``cube`` body from ``robot.mj_data`` (public property) plus the cube's contacts.

Baselines share LayaPolicy's primitive vocabulary (one joint, +/-, small/medium/large, gripper, hold):
- ScriptedPrimitivePolicy: greedy one-step FK lookahead over the 10 arm primitives that most reduces the
  fingertip-to-cube distance; near the cube it closes the gripper, then lifts (pick task).
- RandomPrimitivePolicy: uniform over the same primitives, seeded.
Both log the same per-tick record shape as LayaPolicy.last_tick so the runner treats every arm alike.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from strands_robots import Robot  # noqa: E402
from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.policies.laya.primitives import (  # noqa: E402
    DEFAULT_STEP_DEG,
    GRIPPER_JOINT,
    HOLD,
    SIZE_LABELS,
    Primitive,
    apply_primitive,
    gripper_pct_from_rad,
)
from strands_robots.policies.laya.state_text import labels_for_keys, serialize_state  # noqa: E402

FPS = 10
KEYS = ["1", "2", "3", "4", "5", "6"]
LABELS = labels_for_keys(KEYS)
ARM = tuple(label for label in LABELS.values() if label != GRIPPER_JOINT)
# F3A lane finding (JOURNAL it1-2): rest = (0, -94, 85, 70, 0) deg, gripper 5 % of (-0.1745, 1.7453) rad.
GRIPPER_RANGE = (-0.1745, 1.7453)
REST_QPOS = [
    0.0,
    math.radians(-94.0),
    math.radians(85.0),
    math.radians(70.0),
    0.0,
    GRIPPER_RANGE[0] + 0.05 * (GRIPPER_RANGE[1] - GRIPPER_RANGE[0]),
]
CUBE_SIZE = 0.02
TASKS = {
    "reach": "move the gripper fingertips to the red cube",
    "pick": "pick up the red cube and lift it",
}
REACH_SUCCESS_M = 0.03  # fingertip site within 3 cm of the cube centre
PICK_LIFT_M = 0.05  # cube centre lifted >= 5 cm above the table


def build(cam_size: tuple[int, int] = (256, 256), cameras: bool = True) -> Robot:
    robot = Robot("so101", mode="sim")
    w, h = cam_size
    if cameras:
        assert (
            robot.add_camera(name="scene", position=[0.30, -0.62, 0.32], target=[0.0, -0.15, 0.06], width=w, height=h)[
                "status"
            ]
            == "success"
        )
        assert (
            robot.add_camera(
                name="wrist",
                parent_body="so101/gripper",
                position=[0.06, 0.0, 0.0],
                target=[-0.008, 0.0, -0.16],
                width=w,
                height=h,
            )["status"]
            == "success"
        )
    assert (
        robot.add_object(
            name="cube",
            shape="box",
            position=[0.0, -0.22, CUBE_SIZE / 2],
            size=[CUBE_SIZE] * 3,
            color=[0.9, 0.1, 0.1, 1],
            mass=0.03,
        )["status"]
        == "success"
    )
    rec = robot._world.robots["so101"]
    rec.home_qpos = {f"so101/{i + 1}": [REST_QPOS[i]] for i in range(6)}
    rec.home_actuators = {f"so101/{i + 1}": (REST_QPOS[i], []) for i in range(6)}
    robot.reset()
    return robot


def randomized_cube_xy(rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.28, -0.18)))


class World:
    """Privileged reader over the live MjModel/MjData of a built scene."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        m = robot.mj_model
        self.model = m
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "so101/gripper")
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.cube}
        self.finger_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "so101/so101/static_finger"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "so101/so101/moving_finger"),
        }
        self.jnt_qposadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"so101/{k}")] for k in KEYS]
        self._scratch = mujoco.MjData(m)
        self.ctrl_bounds = {
            k: (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
            for k, j in ((k, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"so101/{k}")) for k in KEYS)
        }
        assert self.site >= 0 and self.cube >= 0 and -1 not in self.finger_geoms

    @property
    def data(self) -> Any:
        return self.robot.mj_data

    def gripper_xyz(self) -> list[float]:
        return [float(v) for v in self.data.site_xpos[self.site]]

    def cube_xyz(self) -> list[float]:
        return [float(v) for v in self.data.xpos[self.cube]]

    def contacts(self) -> list[str]:
        d, m = self.data, self.model
        out: list[str] = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.cube_geoms or g2 in self.cube_geoms:
                other = g2 if g1 in self.cube_geoms else g1
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])
                )
                out.append(f"cube:{(name or str(other)).split('/')[-1]}")
        return sorted(set(out))

    def distance(self) -> float:
        g, c = self.gripper_xyz(), self.cube_xyz()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(g, c, strict=True)))

    def finger_contact(self) -> bool:
        return any(c.endswith("finger") for c in self.contacts())

    def read(self) -> dict[str, Any]:
        return {"gripper_xyz_m": self.gripper_xyz(), "cube_xyz_m": self.cube_xyz(), "contacts": self.contacts()}

    def predict_site(self, qpos_by_key: dict[str, float]) -> np.ndarray:
        """Fingertip site position at a hypothetical joint configuration (FK on a scratch MjData)."""
        d = self._scratch
        d.qpos[:] = self.data.qpos
        for k, adr in zip(KEYS, self.jnt_qposadr, strict=True):
            if k in qpos_by_key:
                d.qpos[adr] = qpos_by_key[k]
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site].copy()

    def success(self, task: str) -> bool:
        if task == "reach":
            return self.distance() < REACH_SUCCESS_M
        return self.cube_xyz()[2] > PICK_LIFT_M and self.finger_contact()


def _tick_record(
    tick: int, primitive: Primitive, latency_ms: float, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    rec = {
        "tick": tick,
        "primitive": primitive.as_dict(),
        "applied": primitive.as_dict(),
        "confidences": {},
        "gate_value": None,
        "gated": False,
        "latency_ms": latency_ms,
    }
    if extra:
        rec.update(extra)
    return rec


class _PrimitivePolicy(Policy):
    """Common plumbing: reads joint state, applies one primitive per tick, records last_tick."""

    reads_instruction = True

    def __init__(self, world: World) -> None:
        self.world = world
        self.robot_state_keys = list(KEYS)
        self.last_tick: dict[str, Any] | None = None
        self.tick_index = 0
        self.instruction = ""

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = list(robot_state_keys) or list(KEYS)

    def reset(self, seed: int | None = None) -> None:
        self.last_tick = None
        self.tick_index = 0

    def _current(self, obs: dict[str, Any]) -> dict[str, float]:
        return {k: float(obs[k]) for k in KEYS}

    def _emit(
        self, primitive: Primitive, current: dict[str, float], t0: float, extra: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        action = apply_primitive(
            primitive, current, LABELS, gripper_range=GRIPPER_RANGE, ctrl_bounds=self.world.ctrl_bounds
        )
        # the same text Laya would have read at this tick, so a judge can score this proposal offline
        state = serialize_state(
            instruction=self.instruction,
            joints_rad=current,
            labels=LABELS,
            gripper_pct=gripper_pct_from_rad(current["6"], GRIPPER_RANGE),
            world=self.world.read(),
        )
        self.last_tick = _tick_record(
            self.tick_index, primitive, (time.perf_counter() - t0) * 1000.0, {**(extra or {}), "state": state}
        )
        self.tick_index += 1
        return [action]


class RandomPrimitivePolicy(_PrimitivePolicy):
    provider_name = "random"

    def __init__(self, world: World, seed: int = 0) -> None:
        super().__init__(world)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        self.instruction = instruction
        joint = str(self.rng.choice(list(ARM) + [GRIPPER_JOINT, "none"]))
        direction = float(self.rng.choice([1.0, -1.0]))
        size = str(self.rng.choice(list(SIZE_LABELS)))
        return self._emit(Primitive(joint, direction, size), self._current(observation_dict), t0)


class ScriptedPrimitivePolicy(_PrimitivePolicy):
    """Greedy FK lookahead over the shared primitives; grasp + lift phase for the pick task."""

    provider_name = "scripted"

    def __init__(self, world: World, task: str) -> None:
        super().__init__(world)
        self.task = task
        self.phase = "approach"
        self.close_ticks = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.phase = "approach"
        self.close_ticks = 0

    def _best_arm_primitive(
        self, current: dict[str, float], size: str, target: np.ndarray, maximize_z: bool = False
    ) -> tuple[Primitive, float]:
        best, best_score = HOLD, None
        step = math.radians(DEFAULT_STEP_DEG[size])
        for key, label in LABELS.items():
            if label == GRIPPER_JOINT:
                continue
            for sign in (1.0, -1.0):
                trial = dict(current)
                lo, hi = self.world.ctrl_bounds[key]
                trial[key] = min(max(current[key] + sign * step, lo), hi)
                site = self.world.predict_site(trial)
                score = -float(site[2]) if maximize_z else float(np.linalg.norm(site - target))
                if best_score is None or score < best_score:
                    best, best_score = Primitive(label, sign, size), score
        return best, float(best_score if best_score is not None else 0.0)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        self.instruction = instruction
        current = self._current(observation_dict)
        cube = np.array(self.world.cube_xyz())
        dist = self.world.distance()
        open_pct = gripper_pct_from_rad(current["6"], GRIPPER_RANGE)
        extra = {"phase": self.phase, "distance_before": dist}

        def approach(target: np.ndarray, d: float) -> Primitive:
            size = "large" if d > 0.08 else "medium" if d > 0.03 else "small"
            prim, predicted = self._best_arm_primitive(current, size, target)
            if predicted >= d - 1e-4 and size != "small":
                prim, predicted = self._best_arm_primitive(current, "small", target)
            extra["predicted_distance"] = predicted
            return prim

        if self.task == "reach":
            return self._emit(approach(cube, dist), current, t0, extra)
        # pick: open -> above -> descend -> grasp -> lift
        if self.phase == "approach":
            if open_pct < 80.0:
                return self._emit(Primitive(GRIPPER_JOINT, 1.0, "large"), current, t0, extra)
            above = cube + np.array([0.0, 0.0, 0.05])
            d_above = float(np.linalg.norm(np.array(self.world.gripper_xyz()) - above))
            if d_above < 0.015:
                self.phase = "descend"
            else:
                return self._emit(approach(above, d_above), current, t0, extra)
        if self.phase == "descend":
            target = cube + np.array([0.0, 0.0, -0.008])
            d_t = float(np.linalg.norm(np.array(self.world.gripper_xyz()) - target))
            if d_t < 0.012 or (self.world.finger_contact() and d_t < 0.02):
                self.phase = "grasp"
            else:
                return self._emit(approach(target, d_t), current, t0, extra)
        if self.phase == "grasp":
            self.close_ticks += 1
            if open_pct <= 1.0 or self.close_ticks > 12:
                self.phase = "lift"
            return self._emit(Primitive(GRIPPER_JOINT, -1.0, "large"), current, t0, extra)
        prim, _ = self._best_arm_primitive(current, "medium", cube, maximize_z=True)
        return self._emit(prim, current, t0, extra)
