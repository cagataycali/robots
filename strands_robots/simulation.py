"""One world. Owns time. Holds many robots.

``Simulation`` is backend-agnostic: it owns the ``Scene`` (plain data), the
robot registry of the world and the time policy. A ``Backend`` implements
a dozen methods and nothing else; MuJoCo is the first. Adding a robot = append to
the Scene + ``backend.load(scene)`` with joint state carried over for the bodies
that still exist.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np

from .description import Description
from .types import Contact, JointState, Physics, Pose

_BACKENDS: dict[str, str] = {
    "mujoco": "strands_robots.backends.mujoco:MuJoCoBackend",
}
_PHYSICS = ("native",)
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _qmul(_qmul(q, np.r_[0.0, v]), q * [1, -1, -1, -1])[1:]


def _moved_frame(frame: Pose, old: Pose, new: Pose) -> Pose:
    """The spawn frame that puts a body at ``new`` when it was at ``old`` under ``frame``: new * old^-1 * frame."""
    q = _qmul(new.orientation, old.orientation * [1, -1, -1, -1])
    q = q / np.linalg.norm(q)
    return Pose(new.position + _rotate(q, frame.position - old.position), _qmul(q, frame.orientation))


class Member(Protocol):
    """What a world needs from a robot it holds: an id, a Description, a hook for when it is removed.

    ``Robot`` is the one implementation; the world asks for nothing more, so it never imports it.
    """

    id: str
    description: Description

    def _detached(self) -> None: ...


# scene: plain data ---------------------------------------------------------------------------------


@dataclass
class SceneRobot:
    id: str
    description: Description
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    keyframe: str | None = None


@dataclass
class SceneObject:
    name: str
    shape: str | None = None  # box | sphere | cylinder | capsule
    path: str | None = None  # a mesh/MJCF/URDF file instead of a primitive
    size: tuple[float, ...] = (0.05, 0.05, 0.05)
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    mass: float | None = None
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    free: bool = True


@dataclass
class SceneCamera:
    name: str
    position: tuple[float, float, float] = (1.5, -1.5, 1.2)
    target: tuple[float, float, float] | None = (0.0, 0.0, 0.3)
    orientation: tuple[float, float, float, float] | None = None
    fov: float = 45.0
    size: tuple[int, int] = (640, 480)
    attach: str | None = None  # robot id whose base body carries the camera


@dataclass
class Scene:
    """Everything a backend needs to build the world. No engine objects inside."""

    robots: list[SceneRobot] = field(default_factory=list)
    objects: list[SceneObject] = field(default_factory=list)
    cameras: list[SceneCamera] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    ground: bool = True
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    light_scale: float = 1.0

    def robot(self, id: str) -> SceneRobot:
        for r in self.robots:
            if r.id == id:
                return r
        raise KeyError(f"no robot {id!r} in scene; robots are {[r.id for r in self.robots]}")


class Object:
    """A handle to a scene object: ``.pose`` / ``.set_pose`` like a Robot."""

    def __init__(self, sim: Simulation, spec: SceneObject) -> None:
        self.sim = sim
        self.spec = spec
        self.name = spec.name

    @property
    def pose(self) -> Pose:
        return self.sim.backend.body_pose(self.name)

    def set_pose(self, position: Any = None, orientation: Any = None) -> None:
        self.sim.set_pose(self.name, position=position, orientation=orientation)

    def __repr__(self) -> str:
        return f"Object({self.name!r}, {self.spec.shape or self.spec.path})"


# backend contract ----------------------------------------------------------------------------------


class Backend(ABC):
    """The twelve methods a physics world implements. Nothing else is engine-specific."""

    def __init__(
        self,
        *,
        physics: str = "native",
        headless: bool = True,
        device: str | None = None,
        num_envs: int = 1,
        render_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.physics_name = physics
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.render_size = render_size

    @abstractmethod
    def load(self, scene: Scene) -> None: ...  # (re)build the world; keep joint state of surviving bodies

    @abstractmethod
    def step(self, n: int) -> None: ...

    @abstractmethod
    def reset(self, keyframe: str | None) -> None: ...

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def joint_state(self, body: str) -> JointState: ...

    @abstractmethod
    def set_targets(self, body: str, targets: np.ndarray) -> None: ...

    @abstractmethod
    def body_pose(self, body: str) -> Pose: ...

    @abstractmethod
    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None: ...  # teleport

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        """Actuator PD gains for one robot (``action_keys`` order). Engines that cannot say so."""
        raise NotImplementedError(f"{type(self).__name__} cannot change actuator gains at runtime")

    @abstractmethod
    def render(self, camera: str) -> np.ndarray: ...

    @abstractmethod
    def contacts(self) -> list[Contact]: ...

    @abstractmethod
    def physics(self) -> Physics: ...

    @abstractmethod
    def close(self) -> None: ...


def _load_backend(name: str) -> type[Backend]:
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown simulation backend {name!r}; choose from {sorted(_BACKENDS)}") from None
    mod_name, cls_name = target.split(":")
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)


# the world -----------------------------------------------------------------------------------------


class Simulation:
    """One world. Owns time. Holds many robots.

    ``Simulation()`` is MuJoCo, headless. ``Simulation.default()`` is the process-wide world
    every Robot joins when it is given none.
    """

    _default: ClassVar[Simulation | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        backend: str = "mujoco",
        *,
        physics: str = "native",
        timestep: float = 0.002,
        substeps: int = 10,
        headless: bool = True,
        device: str | None = None,
        num_envs: int = 1,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        ground: bool = True,
        render_size: tuple[int, int] = (640, 480),
    ) -> None:
        if physics not in _PHYSICS:
            raise ValueError(f"physics={physics!r}; choose from {_PHYSICS}")
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if num_envs > 1:
            raise ValueError(f"num_envs={num_envs} needs vectorised physics; {physics!r} steps one world")
        self.backend_name = backend
        self.physics_name = physics
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.render_size = render_size
        self.scene = Scene(gravity=gravity, timestep=timestep, ground=ground)
        self._substeps = substeps
        self._robots: dict[str, Member] = {}
        self.revision = 0  # bumped on every scene change (add/remove/add_object/camera/set_pose/randomize)
        self.lock = threading.RLock()  # one lock around the world: agents' act() and a dashboard's reads share it
        self._manual = 0
        self._closed = False
        cls = _load_backend(backend)
        self.backend: Backend = cls(
            physics=physics, headless=headless, device=device, num_envs=num_envs, render_size=render_size
        )
        self.backend.load(self.scene)

    # default world

    @classmethod
    def default(cls) -> Simulation:
        """The world a Robot attaches to when none is given: MuJoCo, headless, created on first use."""
        with cls._lock:
            if cls._default is None or cls._default._closed:
                cls._default = cls()
            return cls._default

    @property
    def is_default(self) -> bool:
        return Simulation._default is self

    # time

    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps (``timestep`` each)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        with self.lock:
            self.backend.step(int(n))

    def control_step(self) -> None:
        """Advance exactly one control step (``physics.substeps`` physics steps) unless time is manual."""
        if not self._manual:
            with self.lock:
                self.backend.step(self._substeps)

    @contextmanager
    def manual_time(self) -> Iterator[Simulation]:
        """Inside: ``robot.act()`` only writes targets; the caller steps the world."""
        self._manual += 1
        try:
            yield self
        finally:
            self._manual -= 1

    @property
    def time_is_manual(self) -> bool:
        return self._manual > 0

    def reset(self, *, keyframe: str | None = None, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        with self.lock:
            self.backend.reset(keyframe)

    @property
    def time(self) -> float:
        return self.backend.time()

    @property
    def physics(self) -> Physics:
        p = self.backend.physics()
        return Physics(p.name, p.timestep, self._substeps)

    # scene

    @property
    def robots(self) -> Mapping[str, Member]:
        return dict(self._robots)

    @property
    def envs(self) -> int:
        return self.num_envs

    def _unique_id(self, want: str) -> str:
        if want not in self._robots:
            return want
        n = 2
        while f"{want}_{n}" in self._robots:
            n += 1
        return f"{want}_{n}"

    def add(
        self,
        robot: Member,
        *,
        position: Any = None,
        orientation: Any = None,
        keyframe: str | None = None,
    ) -> Member:
        """Put ``robot`` into this world (called by ``Robot.__init__``). Surviving bodies keep their state."""
        if robot.id in self._robots:
            raise ValueError(f"robot id {robot.id!r} is already in this world")
        pos = tuple(float(x) for x in (position if position is not None else robot.description.spawn_position))
        quat = tuple(float(x) for x in (orientation if orientation is not None else _IDENTITY_QUAT))
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError("position must have 3 values and orientation 4 (w, x, y, z)")
        if keyframe is not None and keyframe not in robot.description.keyframes:
            raise ValueError(f"keyframe {keyframe!r} not in {robot.description.name}: {robot.description.keyframes}")
        with self.lock:  # scene, backend and registry change together: a locked reader never sees them disagree
            self.scene.robots.append(SceneRobot(robot.id, robot.description, pos, quat, keyframe))  # type: ignore[arg-type]
            try:
                self._reload()
            except Exception:
                self.scene.robots.pop()
                raise
            self._robots[robot.id] = robot
        return robot

    def _reload(self) -> None:
        with self.lock:
            self.backend.load(self.scene)
            self.revision += 1

    def remove(self, robot: Member | str) -> None:
        rid = robot if isinstance(robot, str) else robot.id
        if rid not in self._robots:
            raise KeyError(f"no robot {rid!r} in this world; robots are {list(self._robots)}")
        with self.lock:
            self.scene.robots = [r for r in self.scene.robots if r.id != rid]
            self.scene.cameras = [c for c in self.scene.cameras if c.attach != rid]
            self._reload()
            gone = self._robots.pop(rid)
        gone._detached()

    def add_object(
        self,
        name: str,
        *,
        shape: str | None = None,
        path: str | None = None,
        position: Any = (0.0, 0.0, 0.5),
        orientation: Any = None,
        size: Any = (0.05, 0.05, 0.05),
        mass: float | None = None,
        color: Any = None,
        free: bool = True,
    ) -> Object:
        if (shape is None) == (path is None):
            raise ValueError("add_object needs exactly one of shape= or path=")
        if shape is not None and shape not in ("box", "sphere", "cylinder", "capsule"):
            raise ValueError(f"shape={shape!r}; choose from box, sphere, cylinder, capsule")
        if any(o.name == name for o in self.scene.objects) or name in self._robots:
            raise ValueError(f"name {name!r} is already used in this world")
        obj = SceneObject(
            name=name,
            shape=shape,
            path=path,
            size=tuple(float(x) for x in np.atleast_1d(size)),
            position=tuple(float(x) for x in position),  # type: ignore[arg-type]
            orientation=tuple(float(x) for x in (orientation or _IDENTITY_QUAT)),  # type: ignore[arg-type]
            mass=mass,
            color=tuple(float(x) for x in color) if color is not None else (0.6, 0.6, 0.6, 1.0),  # type: ignore[arg-type]
            free=free,
        )
        with self.lock:
            self.scene.objects.append(obj)
            try:
                self._reload()
            except Exception:
                self.scene.objects.pop()
                raise
        return Object(self, obj)

    def set_pose(self, body: Member | Object | str, position: Any = None, orientation: Any = None) -> None:
        """Teleport a robot or object: new base pose, velocities zeroed. Time does not advance.

        A fixed base is rebuilt where it lands and stays there. A floating base keeps its spawn
        frame, so ``reset()`` stands it back on its keyframe instead of adding the keyframe's
        height on top of the teleport.
        """
        name = body if isinstance(body, str) else (body.id if hasattr(body, "id") else body.name)
        if name not in self._robots and not any(o.name == name for o in self.scene.objects):
            raise KeyError(f"no body {name!r} in this world")
        with self.lock:
            cur = self.backend.body_pose(name)
            pose = Pose(
                cur.position if position is None else np.asarray(position, dtype=float),
                cur.orientation if orientation is None else np.asarray(orientation, dtype=float),
            )
            self.backend.set_pose(name, pose, zero_velocity=True)
            self.revision += 1
            if name in self._robots:
                if self._robots[name].description.floating:
                    return
                sr = self.scene.robot(name)  # the next reload rebuilds the fixed base where it now is
                f = _moved_frame(Pose(np.asarray(sr.position), np.asarray(sr.orientation)), cur, pose)
                sr.position, sr.orientation = tuple(f.position.tolist()), tuple(f.orientation.tolist())  # type: ignore[assignment]
            else:
                for o in self.scene.objects:
                    if o.name == name:
                        o.position, o.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())  # type: ignore[assignment]

    def camera(
        self,
        name: str,
        *,
        position: Any = (1.5, -1.5, 1.2),
        target: Any = (0.0, 0.0, 0.3),
        orientation: Any = None,
        fov: float = 45.0,
        size: tuple[int, int] | None = None,
        attach: Member | str | None = None,
    ) -> SceneCamera:
        if any(c.name == name for c in self.scene.cameras):
            raise ValueError(f"camera {name!r} already exists")
        att = attach if (attach is None or isinstance(attach, str)) else attach.id
        if att is not None and att not in self._robots:
            raise KeyError(f"attach={att!r} is not a robot in this world")
        cam = SceneCamera(
            name=name,
            position=tuple(float(x) for x in position),  # type: ignore[arg-type]
            target=None if orientation is not None else tuple(float(x) for x in target),  # type: ignore[arg-type]
            orientation=None if orientation is None else tuple(float(x) for x in orientation),  # type: ignore[arg-type]
            fov=float(fov),
            size=tuple(size) if size else self.render_size,  # type: ignore[arg-type]
            attach=att,
        )
        with self.lock:
            self.scene.cameras.append(cam)
            try:
                self._reload()
            except Exception:
                self.scene.cameras.pop()
                raise
        return cam

    def render(self, camera: str | SceneCamera | None = None) -> np.ndarray:
        """HxWx3 uint8 from a named camera, or from an auto-framed overview camera when none is given."""
        if camera is None:  # the auto-framed overview, created on first use (user cameras never replace it)
            if not any(c.name == "overview" for c in self.scene.cameras):
                self.camera("overview", **self._frame_all())
            name = "overview"
        else:
            name = camera if isinstance(camera, str) else camera.name
        assert name is not None
        if not any(c.name == name for c in self.scene.cameras):
            raise KeyError(f"no camera {name!r}; cameras are {[c.name for c in self.scene.cameras]}")
        with self.lock:
            return self.backend.render(name)

    def _frame_all(self) -> dict[str, Any]:
        """A camera position that sees every robot: look at their centroid from the -y side."""
        pts = np.array([r.position for r in self.scene.robots] or [(0.0, 0.0, 0.0)])
        center = pts.mean(axis=0) + np.array([0.0, 0.0, 0.4])
        spread = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
        dist = max(1.8, 1.4 * spread + 1.6)
        return {"position": tuple(center + np.array([0.0, -dist, 0.65 * dist])), "target": tuple(center)}

    def randomize(
        self,
        *,
        friction: float | tuple[float, float] | None = None,
        mass: float | tuple[float, float] | None = None,
        lighting: float | tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Scale friction / mass / light intensity (a factor or a (lo, hi) range to sample) and rebuild."""
        rng = np.random.default_rng(seed)

        def pick(v: float | tuple[float, float] | None, current: float) -> float:
            if v is None:
                return current
            if isinstance(v, int | float):
                return float(v)
            return float(rng.uniform(*v))

        with self.lock:
            self.scene.friction_scale = pick(friction, self.scene.friction_scale)
            self.scene.mass_scale = pick(mass, self.scene.mass_scale)
            self.scene.light_scale = pick(lighting, self.scene.light_scale)
            self._reload()

    def contacts(self) -> list[Contact]:
        return self.backend.contacts()

    # agents

    def describe(self) -> dict[str, Any]:
        """Robots, objects, cameras, physics, time — JSON-safe; the dashboard reads THIS (under the world lock)."""
        with self.lock:
            return self._describe()

    def _describe(self) -> dict[str, Any]:
        p = self.physics
        return {
            "backend": self.backend_name,
            "physics": {"name": p.name, "timestep": p.timestep, "substeps": p.substeps, "envs": self.num_envs},
            "time": self.time,
            "revision": self.revision,
            "manual_time": self.time_is_manual,
            "robots": {
                rid: r.description.describe() | {"position": list(self.scene.robot(rid).position)}
                for rid, r in self._robots.items()
            },
            "objects": [
                {"name": o.name, "shape": o.shape or o.path, "position": list(o.position)} for o in self.scene.objects
            ],
            "cameras": [{"name": c.name, "size": list(c.size), "attach": c.attach} for c in self.scene.cameras],
        }

    # lifecycle

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in list(self._robots.values()):
            r._detached()
        self._robots.clear()
        self.backend.close()
        if Simulation._default is self:
            Simulation._default = None

    def __enter__(self) -> Simulation:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Simulation({self.backend_name!r}, physics={self.physics_name!r}, "
            f"robots={list(self._robots)}, t={self.time:.3f})"
        )
