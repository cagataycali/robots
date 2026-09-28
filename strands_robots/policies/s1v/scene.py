"""The MuJoCo ``so101`` tabletop rig S1V is trained and evaluated on.

One scene camera and one wrist camera (the same placement the ``flux3_action``
lane uses, so recordings from the three System 1 / System 2 lanes look alike),
a 2 cm red cube on the ground plane, and *privileged* accessors (tool centre
point, cube pose, finger contacts, joint-limit and table hits) that only the
scripted expert and the labeller read. The learned policy never sees any of
them: it gets the two images and the six joint values.

Everything here talks to :class:`~strands_robots.simulation.mujoco.MuJoCoSimEngine`
through ``Robot("so101", mode="sim")`` plus the engine's raw ``mjModel`` /
``mjData`` for the privileged reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strands_robots.policies.s1v.primitives import SO101_CTRL_BOUNDS, SO101_REST_QPOS_RAD

ROBOT = "so101"
CUBE = "cube"
#: The object is a 2 x 2 x 4 cm standing block ("cube" in the code and dataset
#: for continuity). A true 2 cm cube cannot be grasped by this gripper model
#: without the static fingertip touching the table: the moving pad closes
#: ~1 cm shallower than the static pad, so with the cube centre at 1 cm the
#: moving pad passes over the top. Doubling the height puts the grasp band at
#: 2-3 cm with the fingertip ~1.5 cm clear of the table.
CUBE_HALF = 0.01
CUBE_HALF_Z = 0.02
SCENE_CAMERA: dict[str, Any] = {"name": "scene", "position": [0.30, -0.62, 0.32], "target": [0.0, -0.15, 0.06]}
WRIST_CAMERA: dict[str, Any] = {
    "name": "wrist",
    "parent_body": "so101/gripper",
    "position": [0.06, 0.0, 0.0],
    "target": [-0.008, 0.0, -0.16],
}
STATIC_FINGER = "so101/so101/static_finger"
MOVING_FINGER = "so101/so101/moving_finger"
GROUND = "ground"
PAD_HALF_LENGTH = 0.02
PAD_THICKNESS = 0.003
#: Extra lateral room between the static pad and the cube at the grasp pose; the
#: moving jaw pushes the cube the last few millimetres onto the static pad.
GRASP_CLEARANCE = 0.006
#: Where the cube may spawn: a box in front of the arm that the folded rest
#: posture can reach without a shoulder_pan beyond +-45 degrees.
CUBE_XY_RANGE: tuple[tuple[float, float], tuple[float, float]] = ((-0.10, 0.10), (-0.30, -0.16))
DEFAULT_LIGHT_POS = np.array([0.0, -0.3, 1.5])


@dataclass
class Privileged:
    """One snapshot of the state the expert is allowed to read."""

    tcp: np.ndarray
    jaw_axis: np.ndarray
    close_axis: np.ndarray
    pad_rot: np.ndarray
    cube: np.ndarray
    gap: float
    cube_in_jaws: bool
    limit_hit: bool
    table_hit: bool
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(6))


class So101Scene:
    """Build the rig on a ``Robot("so101", mode="sim")`` and expose privileged reads.

    Args:
        robot: ``Robot("so101", mode="sim")``, which IS the MuJoCo engine; its
            ``_world`` is the :class:`~strands_robots.simulation.models.SimWorld`.
        image_size: Camera resolution (square) for both cameras.
        seed: Seed for the per-episode randomisation stream.
    """

    def __init__(self, robot: Any, *, image_size: int = 224, seed: int = 0) -> None:
        self.robot = robot
        self.engine = robot
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size
        self._mj = __import__("mujoco")
        for step in (
            robot.add_camera(width=image_size, height=image_size, **SCENE_CAMERA),
            robot.add_camera(width=image_size, height=image_size, **WRIST_CAMERA),
            robot.add_object(
                name=CUBE,
                shape="box",
                position=[0.0, -0.20, CUBE_HALF_Z],
                size=[CUBE_HALF * 2, CUBE_HALF * 2, CUBE_HALF_Z * 2],
                color=[0.9, 0.1, 0.1, 1],
                mass=0.03,
            ),
        ):
            if step["status"] != "success":
                raise RuntimeError(f"So101Scene: scene step refused: {step['content']}")
        self._park_at_rest()
        self._resolve_ids()
        self._grasp_offset = self._closed_pad_offset()
        self._light0 = self.model.light_pos[0].copy() if self.model.nlight else DEFAULT_LIGHT_POS.copy()

    # -- construction -------------------------------------------------------
    @property
    def model(self) -> Any:
        return self.engine._world._model

    @property
    def data(self) -> Any:
        return self.engine._world._data

    def _park_at_rest(self) -> None:
        record = self.engine._world.robots[ROBOT]
        record.home_qpos = {f"{ROBOT}/{i + 1}": [q] for i, q in enumerate(SO101_REST_QPOS_RAD)}
        record.home_actuators = {f"{ROBOT}/{i + 1}": (q, []) for i, q in enumerate(SO101_REST_QPOS_RAD)}
        self.robot.reset()

    def _resolve_ids(self) -> None:
        mj, m = self._mj, self.model
        self.cube_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, CUBE)
        self.cube_geom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.static_finger = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, STATIC_FINGER)
        self.moving_finger = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, MOVING_FINGER)
        self.ground = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, GROUND)
        self.gripper_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "so101/gripper")
        self.joint_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{ROBOT}/{i}") for i in range(1, 7)]
        self.qpos_adr = [int(m.jnt_qposadr[j]) for j in self.joint_ids]
        self.dof_adr = [int(m.jnt_dofadr[j]) for j in self.joint_ids]
        self.finger_geoms = {self.static_finger, self.moving_finger}
        robot_bodies = {
            b for b in range(m.nbody) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or "").startswith(ROBOT)
        }
        self.arm_geoms = {
            g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot_bodies and g not in self.finger_geoms
        }

    def _closed_pad_offset(self) -> np.ndarray:
        """Moving-pad centre relative to the static pad, in the static pad's frame, jaws closed.

        The so101 gripper is a single moving jaw hinged against a static one,
        so the two pad geoms are NOT symmetric about the jaw axis; where the
        moving pad lands when the jaw closes is where a grasped cube sits.
        Computed once at construction with the gripper forced closed.
        """
        mj, m, d = self._mj, self.model, self.data
        saved = float(d.qpos[self.qpos_adr[5]])
        d.qpos[self.qpos_adr[5]] = m.jnt_range[self.joint_ids[5]][0]
        mj.mj_forward(m, d)
        rot = np.array(d.geom_xmat[self.static_finger]).reshape(3, 3)
        offset = rot.T @ (d.geom_xpos[self.moving_finger] - d.geom_xpos[self.static_finger])
        d.qpos[self.qpos_adr[5]] = saved
        mj.mj_forward(m, d)
        return offset

    # -- episode control ----------------------------------------------------
    def reset_episode(self, *, randomize: bool = True) -> dict[str, Any]:
        """Rest posture, new cube pose, jittered light and scene camera.

        Returns:
            The episode's sampled parameters (for the dataset's episode metadata).
        """
        self.robot.reset()
        if randomize:
            x = float(self.rng.uniform(*CUBE_XY_RANGE[0]))
            y = float(self.rng.uniform(*CUBE_XY_RANGE[1]))
            yaw = float(self.rng.uniform(-math.pi / 4, math.pi / 4))
        else:
            x, y, yaw = 0.0, -0.22, 0.0
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        moved = self.engine.move_object(CUBE, position=[x, y, CUBE_HALF_Z], orientation=quat)
        if moved["status"] != "success":
            raise RuntimeError(f"So101Scene: move_object refused: {moved['content']}")
        light = None
        if randomize and self.model.nlight:
            light = self._light0 + self.rng.uniform(-0.4, 0.4, size=3) * np.array([1.0, 1.0, 0.5])
            self.model.light_pos[0] = light
        cam = None
        if randomize:
            cam_id = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, "scene")
            if cam_id >= 0:
                if not hasattr(self, "_cam0"):
                    self._cam0 = self.model.cam_pos[cam_id].copy()
                cam = self._cam0 + self.rng.uniform(-0.015, 0.015, size=3)
                self.model.cam_pos[cam_id] = cam
        self._mj.mj_forward(self.model, self.data)
        self.cube_start = self.cube_pos().copy()
        return {
            "cube_xy_yaw": [x, y, yaw],
            "light_pos": None if light is None else [float(v) for v in light],
            "scene_cam_pos": None if cam is None else [float(v) for v in cam],
        }

    def observation(self) -> dict[str, Any]:
        """Flat observation: ``"1".."6"`` (+ ``.vel``) and the ``scene`` / ``wrist`` images."""
        return self.engine.get_observation(ROBOT)

    def apply(self, action: dict[str, float], *, hz: float = 10.0) -> None:
        """Write joint targets and advance the physics for one control period."""
        n = max(1, int(round(1.0 / (hz * float(self.model.opt.timestep)))))
        self.engine.send_action(action, ROBOT, n_substeps=n)

    # -- privileged reads ---------------------------------------------------
    def qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.qpos_adr], dtype=float)

    def cube_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.cube_body], dtype=float)

    def tcp(self) -> np.ndarray:
        """Tool centre point: where a grasped cube's centre sits, in world coordinates.

        Measured in the static pad's frame: the moving pad closes along the
        static pad's local +x (6.3 cm away at 65 % open, 0.4 cm when closed),
        the fingertip is at local -z (one pad half-length, 2 cm), and the
        gripper body is at +z. A 2 cm cube standing on the table with the
        fingertip touching the table therefore has its centre one cube
        half-size along +x from the pad face and one cube half-size above the
        tip: local ``(CUBE_HALF + pad thickness, 0, -PAD_HALF_LENGTH + CUBE_HALF)``.
        """
        rot = np.array(self.data.geom_xmat[self.static_finger]).reshape(3, 3)
        local = np.array([CUBE_HALF + PAD_THICKNESS + GRASP_CLEARANCE, 0.0, -PAD_HALF_LENGTH + CUBE_HALF])
        return np.array(self.data.geom_xpos[self.static_finger]) + rot @ local

    def jaw_axis(self) -> np.ndarray:
        """World direction the jaws point along (gripper local -z)."""
        return -np.array(self.data.xmat[self.gripper_body]).reshape(3, 3)[:, 2]

    def close_axis(self) -> np.ndarray:
        """World direction the moving pad travels along when the jaw closes (static pad local +x)."""
        return np.array(self.data.geom_xmat[self.static_finger]).reshape(3, 3)[:, 0]

    def finger_gap(self) -> float:
        """Opening between the pads, measured along the static pad's closing axis (metres)."""
        rot = np.array(self.data.geom_xmat[self.static_finger]).reshape(3, 3)
        return float((rot.T @ (self.data.geom_xpos[self.moving_finger] - self.data.geom_xpos[self.static_finger]))[0])

    def tcp_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """``(J_pos, J_rot)`` of the TCP w.r.t. the five arm joints (3 x 5 each)."""
        mj, m, d = self._mj, self.model, self.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jacp, jacr, self.tcp(), self.gripper_body)
        cols = self.dof_adr[:5]
        return jacp[:, cols], jacr[:, cols]

    def contacts(self) -> tuple[bool, bool, bool]:
        """``(cube_in_jaws, limit_hit, table_hit)`` from the current contact set."""
        mj, m, d = self._mj, self.model, self.data
        touching_static = touching_moving = False
        table_hit = False
        for i in range(int(d.ncon)):
            c = d.contact[i]
            if c.exclude != 0 or c.dist > 0.0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if self.cube_geom in pair:
                other = g1 if g2 == self.cube_geom else g2
                touching_static |= other == self.static_finger
                touching_moving |= other == self.moving_finger
            if self.ground in pair and (pair & (self.arm_geoms | self.finger_geoms)):
                table_hit = True  # a fingertip on the table pins the arm too
        q = self.qpos()
        limit_hit = False
        for i, key in enumerate(("1", "2", "3", "4", "5")):
            lo, hi = SO101_CTRL_BOUNDS[key]
            if q[i] <= lo + math.radians(0.5) or q[i] >= hi - math.radians(0.5):
                limit_hit = True
        return touching_static and touching_moving, limit_hit, table_hit

    def pad_frame(self, world_vec: np.ndarray) -> np.ndarray:
        """Express a world vector in the static pad's frame (x = closing axis, y = across the pad, z = along the jaws)."""
        rot = np.array(self.data.geom_xmat[self.static_finger]).reshape(3, 3)
        return rot.T @ world_vec

    def snapshot(self) -> Privileged:
        self._mj.mj_forward(self.model, self.data)
        in_jaws, limit_hit, table_hit = self.contacts()
        return Privileged(
            tcp=self.tcp().copy(),
            jaw_axis=self.jaw_axis().copy(),
            close_axis=self.close_axis().copy(),
            pad_rot=np.array(self.data.geom_xmat[self.static_finger]).reshape(3, 3).copy(),
            cube=self.cube_pos().copy(),
            gap=self.finger_gap(),
            cube_in_jaws=in_jaws,
            limit_hit=limit_hit,
            table_hit=table_hit,
            qpos=self.qpos(),
        )
