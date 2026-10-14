"""The MuJoCo world: one MjSpec built from a Scene, recompiled on every change with state carried over.

Robots are attached with ``spec.attach(robot_spec, prefix="<id>/", frame=...)`` so MuJoCo
namespaces bodies, joints, actuators and keyframes for us.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .._gl import ensure_gl
from ..description import Description
from ..simulation import Backend, Scene, SceneCamera, SceneObject, SceneRobot
from ..types import Contact, JointState, Physics, Pose

ensure_gl()
import mujoco  # noqa: E402

_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


@lru_cache(maxsize=64)
def _robot_spec_xml(path: str) -> str:
    """Load a robot MJCF/URDF once, absolutise its asset paths and keep it as XML (specs are consumed by attach)."""
    spec = mujoco.MjSpec.from_file(path)
    base = Path(spec.modelfiledir or Path(path).parent)
    for assets, sub in ((spec.meshes, spec.meshdir), (spec.hfields, spec.meshdir), (spec.textures, spec.texturedir)):
        for a in assets:
            if a.file and not Path(a.file).is_absolute():
                cand = base / (sub or "") / a.file
                if cand.is_file():
                    a.file = str(cand)
    return spec.to_xml()


def _look_at(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """``xyaxes`` for a camera at ``pos`` looking at ``target`` with +z world as up."""
    fwd = target - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x.tolist(), *y.tolist()]


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
    return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _size(shape: str, size: tuple[float, ...]) -> list[float]:
    s = list(size) + [0.0] * (3 - len(size))
    if shape == "sphere":
        return [s[0], 0, 0]
    if shape in ("cylinder", "capsule"):
        return [s[0], s[1] if len(size) > 1 else s[0], 0]
    return s[:3] if len(size) >= 3 else [s[0]] * 3


class MuJoCoBackend(Backend):
    """Builds the world from a Scene; every ``load`` recompiles and carries joint state by name."""

    def __init__(self, **cfg: Any) -> None:
        super().__init__(**cfg)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.spec: mujoco.MjSpec | None = None
        self._scene: Scene = Scene()
        self._robots: dict[str, Description] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._gl_failed: str | None = None

    # --- build ---------------------------------------------------------------------------------

    def load(self, scene: Scene) -> None:
        carried, t = self._snapshot(), (self.data.time if self.data is not None else 0.0)
        spec = mujoco.MjSpec()
        spec.option.timestep = scene.timestep
        spec.option.gravity = list(scene.gravity)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.nkey = sum(len(r.description.keyframes) for r in scene.robots)  # else attach warns "nkey conflict"
        spec.visual.global_.offwidth = max(self.render_size[0], *(c.size[0] for c in scene.cameras), 640)
        spec.visual.global_.offheight = max(self.render_size[1], *(c.size[1] for c in scene.cameras), 480)
        spec.visual.headlight.diffuse = [0.6 * scene.light_scale] * 3
        spec.visual.headlight.ambient = [0.3 * scene.light_scale] * 3
        if scene.ground:
            spec.worldbody.add_geom(
                name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.75, 0.78, 0.82, 1]
            )
        spec.worldbody.add_light(pos=[0.5, -1.0, 3.0], dir=[-0.2, 0.4, -1.0], diffuse=[0.8 * scene.light_scale] * 3)
        for r in scene.robots:
            self._attach_robot(spec, r)
        for o in scene.objects:
            self._add_object(spec, o)
        for c in scene.cameras:
            self._add_camera(spec, c, scene)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spec.compile()
        if scene.friction_scale != 1.0:
            model.geom_friction[:, 0] *= scene.friction_scale
        if scene.mass_scale != 1.0:
            model.body_mass[:] *= scene.mass_scale
            model.body_inertia[:] *= scene.mass_scale
        data = mujoco.MjData(model)
        self.spec, self.model, self.data = spec, model, data
        self._scene = scene
        self._robots = {r.id: r.description for r in scene.robots}
        self._renderers.clear()
        self._apply_keyframes(scene)
        self._restore(carried)
        data.time = t  # a scene change is not a reset: world time carries over like joint state
        mujoco.mj_forward(model, data)

    def _attach_robot(self, spec: mujoco.MjSpec, r: SceneRobot) -> None:
        rs = mujoco.MjSpec.from_string(_robot_spec_xml(str(r.description.model_path)))
        for g in [g for g in rs.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:
            rs.delete(g)  # the world owns the floor
        for light in [li for li in rs.lights if li.parent.name == "world"]:
            rs.delete(light)  # and the lighting: a robot's scene light would stack on the world's and wash it out
        frame = spec.worldbody.add_frame(pos=list(r.position), quat=list(r.orientation))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "attach conflict: integrator" — the world's option wins
            spec.attach(rs, prefix=f"{r.id}/", frame=frame)

    def _add_object(self, spec: mujoco.MjSpec, o: SceneObject) -> None:
        body = spec.worldbody.add_body(name=o.name, pos=list(o.position), quat=list(o.orientation))
        if o.free:
            body.add_freejoint(name=f"{o.name}/free")
        if o.path is not None:
            sub = mujoco.MjSpec.from_string(_robot_spec_xml(str(Path(o.path).expanduser().resolve())))
            frame = body.add_frame()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec.attach(sub, prefix=f"{o.name}/", frame=frame)
            return
        kw: dict[str, Any] = {
            "name": f"{o.name}/geom",
            "type": _GEOM[o.shape or "box"],
            "size": _size(o.shape or "box", o.size),
            "rgba": list(o.color),
        }
        if o.mass is not None:
            kw["mass"] = o.mass
        body.add_geom(**kw)

    def _add_camera(self, spec: mujoco.MjSpec, c: SceneCamera, scene: Scene) -> None:
        parent = spec.worldbody
        pos = np.asarray(c.position, dtype=float)
        if c.attach is not None:
            root = self._root_body_name(spec, c.attach)
            parent = spec.body(root)
        kw: dict[str, Any] = {"name": c.name, "pos": pos.tolist(), "fovy": c.fov}
        if c.orientation is not None:
            kw["quat"] = list(c.orientation)
        else:
            tgt = np.asarray(c.target if c.target is not None else (0, 0, 0), dtype=float)
            kw["xyaxes"] = _look_at(pos, tgt)
        parent.add_camera(**kw)

    @staticmethod
    def _root_body_name(spec: mujoco.MjSpec, rid: str) -> str:
        for b in spec.worldbody.bodies:
            if b.name.startswith(f"{rid}/"):
                return b.name
        raise KeyError(f"no attached root body for {rid!r}")

    def _apply_keyframes(self, scene: Scene) -> None:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        for r in scene.robots:
            keys = [r.keyframe] if r.keyframe else list(r.description.keyframes[:1])
            for key in keys:
                kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, f"{r.id}/{key}")
                if kid < 0:
                    continue
                for j in self._joint_ids(r.id):
                    adr, n = m.jnt_qposadr[j], (7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1)
                    d.qpos[adr : adr + n] = m.key_qpos[kid, adr : adr + n]
                    if n == 7:  # attach leaves a keyframe's base pose in the robot frame: move it to the spawn frame
                        fq = np.asarray(r.orientation, dtype=float)
                        d.qpos[adr : adr + 3] = np.asarray(r.position) + _rotate(fq, m.key_qpos[kid, adr : adr + 3])
                        d.qpos[adr + 3 : adr + 7] = _qmul(fq, m.key_qpos[kid, adr + 3 : adr + 7])
                if m.nu:
                    for a in self._actuator_ids(r.id):
                        d.ctrl[a] = m.key_ctrl[kid, a]

    # --- state carry-over -----------------------------------------------------------------------

    def _snapshot(self) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
        if self.model is None or self.data is None:
            return None
        m, d = self.model, self.data
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for j in range(m.njnt):
            jt = m.jnt_type[j]
            nq = 7 if jt == mujoco.mjtJoint.mjJNT_FREE else (4 if jt == mujoco.mjtJoint.mjJNT_BALL else 1)
            nv = 6 if nq == 7 else (3 if nq == 4 else 1)
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            out[m.joint(j).name] = (d.qpos[qa : qa + nq].copy(), d.qvel[va : va + nv].copy())
        out["\x00ctrl"] = (np.array([m.actuator(i).name for i in range(m.nu)], dtype=object), d.ctrl.copy())
        return out

    def _restore(self, carried: dict[str, tuple[np.ndarray, np.ndarray]] | None) -> None:
        if not carried:
            return
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        names, ctrl = carried.pop("\x00ctrl")
        for name, (q, v) in carried.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                continue
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[qa : qa + len(q)] = q
            d.qvel[va : va + len(v)] = v
        for name, c in zip(names, ctrl, strict=True):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
            if a >= 0:
                d.ctrl[a] = c

    # --- lookups ---------------------------------------------------------------------------------

    def _joint_ids(self, rid: str) -> list[int]:
        assert self.model is not None
        m = self.model
        return [j for j in range(m.njnt) if m.joint(j).name.startswith(f"{rid}/")]

    def _actuator_ids(self, rid: str) -> list[int]:
        assert self.model is not None
        m = self.model
        desc = self._robots[rid]
        return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}/{a}") for a in desc.actuators]

    def _body_id(self, name: str) -> int:
        assert self.model is not None
        m = self.model
        if name in self._robots:
            for b in range(1, m.nbody):
                if m.body_parentid[b] == 0 and m.body(b).name.startswith(f"{name}/"):
                    return b
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise KeyError(f"no body {name!r} in the world")
        return b

    # --- the Backend contract ------------------------------------------------------------------

    def step(self, n: int) -> None:
        assert self.model is not None and self.data is not None
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def reset(self, keyframe: str | None) -> None:
        assert self.model is not None and self.data is not None
        mujoco.mj_resetData(self.model, self.data)
        scene = self._scene
        if keyframe is not None:
            scene = Scene(
                **{**vars(scene), "robots": [SceneRobot(**{**vars(r), "keyframe": keyframe}) for r in scene.robots]}
            )
        self._apply_keyframes(scene)
        mujoco.mj_forward(self.model, self.data)

    def time(self) -> float:
        assert self.data is not None
        return float(self.data.time)

    def joint_state(self, body: str) -> JointState:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        desc = self._robots[body]
        # one reading per action key, in actuator order: the driven joint's, or, for an actuator that
        # drives a tendon or a site (panda's gripper), the control that holds its transmission where it is
        # (the servo inverse when the actuator has a position bias, else its control), so that
        # act(observe().joints) is a hold for every key, with the transmission's velocity and force
        pos, vel, eff = (np.empty(len(desc.action_keys)) for _ in range(3))
        for i, (key, a) in enumerate(zip(desc.action_keys, self._actuator_ids(body), strict=True)):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{body}/{desc.unlabel(key)}")
            if j >= 0:
                pos[i], vel[i], eff[i] = (
                    d.qpos[m.jnt_qposadr[j]],
                    d.qvel[m.jnt_dofadr[j]],
                    d.qfrc_actuator[m.jnt_dofadr[j]],
                )
            elif a >= 0:
                gain, bias = m.actuator_gainprm[a, 0], m.actuator_biasprm[a]
                servo = m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and gain != 0 and bias[1] != 0
                pos[i] = -(bias[0] + bias[1] * d.actuator_length[a]) / gain if servo else d.ctrl[a]
                vel[i], eff[i] = d.actuator_velocity[a], d.actuator_force[a]
            else:
                raise KeyError(f"{body}: no joint or actuator for action key {key!r} in the compiled world")
        return JointState(desc.action_keys, pos, vel, eff)

    def set_targets(self, body: str, targets: np.ndarray) -> None:
        assert self.data is not None
        ids = self._actuator_ids(body)
        if len(targets) != len(ids):
            raise ValueError(f"{body}: {len(targets)} targets for {len(ids)} actuators")
        self.data.ctrl[ids] = targets

    def body_pose(self, body: str) -> Pose:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        b = self._body_id(body)
        j = int(m.body_jntadr[b])
        if m.body_jntnum[b] > 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            va = m.jnt_dofadr[j]  # free joint: linear velocity in the world frame, angular in the body frame
            return Pose(d.xpos[b].copy(), d.xquat[b].copy(), d.qvel[va : va + 3].copy(), d.qvel[va + 3 : va + 6].copy())
        return Pose(d.xpos[b].copy(), d.xquat[b].copy())

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        """Make this robot's position actuators the PD law ``tau = kp·(target − q) − kd·q̇`` (no recompile)."""
        assert self.model is not None
        ids = self._actuator_ids(body)
        kp, kd = np.asarray(kp, dtype=float).reshape(-1), np.asarray(kd, dtype=float).reshape(-1)
        if kp.shape != (len(ids),) or kd.shape != (len(ids),):
            raise ValueError(f"{body}: kp {kp.shape} / kd {kd.shape} for {len(ids)} actuators")
        m = self.model
        for i, p, dgain in zip(ids, kp, kd, strict=True):
            m.actuator_gainprm[i, 0] = p
            m.actuator_biasprm[i, 0] = 0.0
            m.actuator_biasprm[i, 1] = -p
            m.actuator_biasprm[i, 2] = -dgain

    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None:
        assert self.model is not None and self.data is not None and self.spec is not None
        m, d = self.model, self.data
        b = self._body_id(body)
        j = int(m.body_jntadr[b])
        if m.body_jntnum[b] > 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[qa : qa + 3] = pose.position
            d.qpos[qa + 3 : qa + 7] = pose.orientation
            if zero_velocity:
                d.qvel[va : va + 6] = 0.0
        else:  # fixed base: the caller has recorded the new pose in the scene; rebuild it the way every change is
            self.load(self._scene)
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str) -> np.ndarray:
        assert self.model is not None and self.data is not None
        if self._gl_failed:
            raise RuntimeError(f"rendering unavailable: {self._gl_failed}")
        cam = next((c for c in self._scene.cameras if c.name == camera), None)
        size = cam.size if cam else self.render_size
        r = self._renderers.get(size)
        if r is None:
            try:
                r = mujoco.Renderer(self.model, height=size[1], width=size[0])
            except Exception as exc:  # noqa: BLE001 — no GL on this host
                self._gl_failed = f"{type(exc).__name__}: {exc} (MUJOCO_GL={ensure_gl() or 'default'})"
                raise RuntimeError(f"rendering unavailable: {self._gl_failed}") from exc
            self._renderers[size] = r
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def contacts(self) -> list[Contact]:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        out: list[Contact] = []
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f)
            out.append(
                Contact(
                    m.body(m.geom_bodyid[c.geom1]).name,
                    m.body(m.geom_bodyid[c.geom2]).name,
                    c.pos.copy(),
                    c.frame[:3].copy(),
                    float(f[0]),
                )
            )
        return out

    def physics(self) -> Physics:
        assert self.model is not None
        return Physics("mujoco", float(self.model.opt.timestep), 1)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        self.model = self.data = self.spec = None

    # --- extras the Simulation may use --------------------------------------------------------------

    def to_xml(self) -> str:
        assert self.spec is not None
        return self.spec.to_xml()

    @staticmethod
    def quat_from_yaw(yaw_deg: float) -> list[float]:
        h = math.radians(yaw_deg) / 2
        return [math.cos(h), 0.0, 0.0, math.sin(h)]
