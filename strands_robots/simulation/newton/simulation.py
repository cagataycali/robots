"""Newton GPU-native simulation backend.

Implements :class:`~strands_robots.simulation.base.SimEngine` on top of
newton-physics/newton (NVIDIA Warp + MuJoCo-Warp). Newton ingests the same
MJCF assets the MuJoCo backend uses (resolved through
:mod:`strands_robots.assets`), builds a GPU model, and steps it with any of
Newton's rigid-body solvers. Rendering is done headlessly via Newton's
ray-traced ``SensorTiledCamera`` so it needs no display server.

The backend reuses the backend-agnostic policy orchestration
(``run_policy`` / ``eval_policy`` / ``replay_episode``) provided by the ABC -
it only implements the abstract physics primitives plus ``render``.

Lifecycle::

    from strands_robots.simulation import create_simulation

    sim = create_simulation("newton", solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")
    sim.send_action({"Rotation": 0.5}, robot_name="so100")
    out = sim.render(width=320, height=240)   # out["image"] -> (H, W, 3) uint8
    sim.destroy()
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.assets import resolve_model_path, resolve_robot_name
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimObject, SimRobot, SimWorld
from strands_robots.simulation.newton.backend import ensure_newton, resolve_solver_class, solver_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Newton's default control rate. MJCF position actuators settle within a few
# hundred substeps; 60 Hz frames with 10 substeps each matches the Newton
# example cadence and keeps position-servo arms tracking their targets.
_DEFAULT_TIMESTEP = 1.0 / 600.0


def _short_joint_name(label: str) -> str:
    """Reduce a hierarchical Newton joint label to its short joint name.

    Newton labels joints by their full body path
    (``so_arm100/worldbody/Base/.../Rotation``); the public observation /
    action schema uses the short trailing segment (``Rotation``) to match the
    MuJoCo backend. The same short name therefore maps to the same joint
    across both backends.

    Args:
        label: Full Newton joint label.

    Returns:
        The trailing path segment.
    """
    return label.rsplit("/", 1)[-1]


class NewtonSimEngine(SimEngine):
    """GPU-native simulation backend built on Newton (Warp / MuJoCo-Warp).

    One Newton model per instance. The world is rebuilt whenever robots or
    objects are added or removed, because Newton finalises an immutable
    ``Model`` from a ``ModelBuilder``. State is preserved across rebuilds
    where joint names still exist.

    Thread-safety: a single ``RLock`` serialises all model/state mutation so a
    ``PolicyRunner`` worker and the calling thread never race on Warp arrays.
    """

    def __init__(
        self,
        solver: str = "mujoco",
        default_timestep: float = _DEFAULT_TIMESTEP,
        substeps: int = 10,
        device: str | None = None,
        default_width: int = 640,
        default_height: int = 480,
        **kwargs: Any,
    ) -> None:
        """Construct a Newton simulation engine.

        Args:
            solver: Friendly solver name. One of
                :func:`~strands_robots.simulation.newton.backend.solver_registry`
                (default ``"mujoco"``, i.e. MuJoCo-Warp).
            default_timestep: Physics integration timestep in seconds.
            substeps: Physics substeps per :meth:`step` call.
            device: Warp device string (e.g. ``"cuda:0"`` or ``"cpu"``).
                ``None`` selects Warp's default device (GPU when available).
            default_width: Default render width in pixels.
            default_height: Default render height in pixels.
            **kwargs: Ignored; accepted for forward compatibility.

        Raises:
            ValueError: If ``solver`` is not a known solver name.
        """
        super().__init__()
        self._nt, self._wp = ensure_newton()
        if solver.lower() not in solver_registry():
            raise ValueError(f"Unknown Newton solver {solver!r}. Available: {sorted(solver_registry())}")
        self._solver_name = solver.lower()
        self.default_timestep = default_timestep
        self.substeps = substeps
        self.device = device
        self.default_width = default_width
        self.default_height = default_height

        self._world: SimWorld | None = None
        self._lock = threading.RLock()

        # Newton handles (rebuilt on every scene mutation via _rebuild).
        self._model: Any = None
        self._solver: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        # Ordered short joint names and their DOF indices in joint_q / target.
        self._joint_order: list[str] = []
        # Pending position targets keyed by (robot_name, short joint name).
        self._targets: dict[tuple[str, str], float] = {}
        # Coordinate index of each (robot, joint) in the global joint_q /
        # joint_target_q vector. For the revolute/prismatic joints of robot
        # arms one coordinate maps to one DOF, so this also indexes targets.
        self._joint_coord_index: dict[tuple[str, str], int] = {}
        # Short joint names per robot (rebuilt with the model).
        self._robot_joint_map: dict[str, list[str]] = {}

        logger.info("Newton simulation engine initialised (solver=%s)", self._solver_name)

    # World lifecycle

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create an empty Newton world.

        Args:
            timestep: Physics timestep in seconds (defaults to the engine's
                ``default_timestep``).
            gravity: Gravity vector ``[x, y, z]`` (default ``[0, 0, -9.81]``).
            ground_plane: Whether to add a ground plane.

        Returns:
            Status dict with a human-readable confirmation.
        """
        with self._lock:
            self._world = SimWorld(
                timestep=timestep or self.default_timestep,
                gravity=gravity or [0.0, 0.0, -9.81],
                ground_plane=ground_plane,
            )
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Newton world created (solver={self._solver_name})."}]}

    def destroy(self) -> dict[str, Any]:
        """Destroy the world and release Newton/Warp handles."""
        with self._lock:
            self._world = None
            self._model = None
            self._solver = None
            self._state_0 = self._state_1 = self._control = None
            self._joint_order = []
            self._targets = {}
        return {"status": "success", "content": [{"text": "Newton world destroyed."}]}

    def reset(self) -> dict[str, Any]:
        """Reset the world to its initial joint configuration."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        with self._lock:
            self._targets = {}
            self._world.sim_time = 0.0
            self._world.step_count = 0
            self._rebuild()
        return {"status": "success", "content": [{"text": "Newton world reset."}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance the simulation by ``n_steps`` control steps."""
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        with self._lock:
            self._advance(n_steps)
        return {"status": "success", "content": [{"text": f"Stepped {n_steps} step(s)."}]}

    def get_state(self) -> dict[str, Any]:
        """Return a human-readable world-state summary."""
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        w = self._world
        lines = [
            "Newton Simulation State",
            f"solver={self._solver_name} t={w.sim_time:.4f}s (step {w.step_count})",
            f"dt={w.timestep}s gravity={w.gravity}",
            f"Robots: {len(w.robots)} | Objects: {len(w.objects)} | DOFs: {self._model.joint_dof_count}",
        ]
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # Robot management

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the world from a registered name or an MJCF path.

        Args:
            name: Robot name in the registry, or an arbitrary instance name
                when ``urdf_path`` points at an explicit MJCF/URDF.
            urdf_path: Optional explicit MJCF/URDF path. When omitted, the
                asset is resolved from the registry by ``name``.
            data_config: Accepted for ABC parity; unused by Newton.
            position: World position ``[x, y, z]`` (default origin).
            orientation: World orientation as a wxyz quaternion
                (default identity).

        Returns:
            Status dict including the resolved joint names.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if name in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' already exists."}]}

        if urdf_path is None:
            try:
                resolved = resolve_robot_name(name)
                asset_path = resolve_model_path(resolved)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                asset_path = None
                _resolve_error: Exception | None = exc
            else:
                _resolve_error = None
            if not asset_path:
                detail = f": {_resolve_error}" if _resolve_error else ""
                return {
                    "status": "error",
                    "content": [
                        {"text": f"Could not resolve a sim asset for robot '{name}'{detail}. See list_robots()."}
                    ],
                }
            model_path = str(asset_path)
        else:
            model_path = urdf_path

        with self._lock:
            robot = SimRobot(
                name=name,
                urdf_path=model_path,
                position=position or [0.0, 0.0, 0.0],
                orientation=orientation or [1.0, 0.0, 0.0, 0.0],
                data_config=data_config,
            )
            self._world.robots[name] = robot
            try:
                self._rebuild()
            except Exception:
                del self._world.robots[name]
                self._rebuild()
                raise
            robot.joint_names = list(self._robot_joint_map.get(name, []))
        return {
            "status": "success",
            "content": [{"text": f"Added robot '{name}' ({len(robot.joint_names)} joints)."}],
        }

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot and rebuild the world."""
        if self._world is None or name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' not found."}]}
        with self._lock:
            del self._world.robots[name]
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Removed robot '{name}'."}]}

    def list_robots(self) -> list[str]:
        """Return the ordered names of robots in the world."""
        if self._world is None:
            return []
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered short joint names for ``robot_name``."""
        if self._world is None or robot_name not in self._world.robots:
            return []
        return list(self._world.robots[robot_name].joint_names)

    # Object management

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        mesh_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a primitive object (box/sphere/capsule/cylinder) to the scene.

        Args:
            name: Unique object name.
            shape: One of ``"box"``, ``"sphere"``, ``"capsule"``,
                ``"cylinder"``.
            position: World position ``[x, y, z]`` (default origin).
            orientation: wxyz quaternion (default identity).
            size: Half-extents (box) or ``[radius, ...]`` (others).
            color: RGBA in 0..1 (alpha currently ignored by Newton shapes).
            mass: Object mass; ``0`` or ``is_static`` makes it static.
            is_static: When True the object is fixed in the world.
            mesh_path: Unused (mesh objects are not yet supported).
            **kwargs: Ignored.

        Returns:
            Status dict.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if name in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' already exists."}]}
        if shape not in ("box", "sphere", "capsule", "cylinder"):
            return {"status": "error", "content": [{"text": f"Unsupported shape {shape!r} for Newton backend."}]}
        with self._lock:
            obj = SimObject(
                name=name,
                shape=shape,
                position=position or [0.0, 0.0, 0.0],
                orientation=orientation or [1.0, 0.0, 0.0, 0.0],
                size=size or [0.05, 0.05, 0.05],
                color=color or [0.5, 0.5, 0.5, 1.0],
                mass=mass,
                is_static=is_static,
            )
            self._world.objects[name] = obj
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Added {shape} object '{name}'."}]}

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object and rebuild the world."""
        if self._world is None or name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        with self._lock:
            del self._world.objects[name]
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Removed object '{name}'."}]}

    # Observation / action

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Return joint positions for a robot keyed by short joint name.

        Args:
            robot_name: Robot to observe. ``None`` resolves to the single
                robot when exactly one exists.
            skip_images: Newton does not attach per-robot cameras here, so this
                flag is a no-op; the observation is joint state only.

        Returns:
            Mapping of short joint name to joint position (float). Empty when
            no world exists or the robot is unknown.
        """
        if self._world is None or self._model is None:
            return {}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError:
            return {}
        if robot_name not in self._world.robots:
            return {}
        with self._lock:
            joint_q = self._state_0.joint_q.numpy()
            robot_joints = self._world.robots[robot_name].joint_names
            obs: dict[str, Any] = {}
            for jname in robot_joints:
                idx = self._joint_coord_index.get((robot_name, jname))
                if idx is not None and idx < len(joint_q):
                    obs[jname] = float(joint_q[idx])
        return obs

    def send_action(self, action: dict[str, Any], robot_name: str | None = None, n_substeps: int = 1) -> dict[str, Any]:
        """Apply position targets and advance physics by ``n_substeps``.

        Args:
            action: Mapping of short joint name to target position (radians).
            robot_name: Robot to actuate. ``None`` resolves to the single
                robot when exactly one exists.
            n_substeps: Number of control steps to advance after writing
                targets.

        Returns:
            Status dict. When some keys cannot be resolved to joints, the
            ``content`` carries a ``json`` block with ``unresolved_keys`` and
            ``applied`` so callers can self-correct.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        valid = set(self._world.robots[robot_name].joint_names)
        unresolved = [k for k in action if k not in valid]
        applied = [k for k in action if k in valid]
        with self._lock:
            for jname in applied:
                self._targets[(robot_name, jname)] = float(action[jname])
            self._write_targets()
            self._advance(n_substeps)

        if unresolved:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action partially applied: keys {unresolved} are not joints on "
                            f"'{robot_name}'. Applied: {applied}. Valid keys: {sorted(valid)}"
                        )
                    },
                    {"json": {"unresolved_keys": unresolved, "applied": applied}},
                ],
            }
        return {"status": "success", "content": [{"text": f"Action applied to '{robot_name}' ({len(applied)} keys)."}]}

    def physics_timestep(self) -> float | None:
        """Return the physics integration timestep in seconds."""
        if self._world is None:
            return None
        return float(self._world.timestep)

    # Rendering

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render the scene headlessly with Newton's ray-traced tiled camera.

        A default three-quarter camera framing the world origin is used. The
        ``camera_name`` argument is accepted for ABC parity; only the default
        view is currently provided.

        Args:
            camera_name: Accepted for ABC parity (only ``"default"`` exists).
            width: Render width in pixels (defaults to ``default_width``).
            height: Render height in pixels (defaults to ``default_height``).

        Returns:
            Agent-tool dict with ``status`` and a ``content`` list. On success
            the content holds an ``image`` block carrying PNG bytes
            (``{"image": {"format": "png", "source": {"bytes": ...}}}``) plus a
            ``json`` block with pixel statistics, matching the MuJoCo backend so
            the shared ``PolicyRunner`` video pipeline consumes it unchanged.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if camera_name not in (None, "", "default", "free"):
            return {
                "status": "error",
                "content": [{"text": f"Camera '{camera_name}' not found. Newton backend only provides 'default'."}],
            }
        w = width or self.default_width
        h = height or self.default_height
        try:
            img = self._render_rgb(w, h)
        except Exception as exc:  # noqa: BLE001 - surface any render failure as a tool error
            return {"status": "error", "content": [{"text": f"Render failed: {exc}"}]}

        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        return {
            "status": "success",
            "content": [
                {"text": f"{w}x{h} from 'default' at t={self._world.sim_time:.3f}s"},
                {"image": {"format": "png", "source": {"bytes": png_bytes}}},
                {
                    "json": {
                        "pixel_variance": float(np.var(img)),
                        "pixel_mean": float(np.mean(img)),
                        "camera": "default",
                    }
                },
            ],
        }

    def _render_rgb(self, w: int, h: int) -> np.ndarray:
        """Render the default view to an ``(H, W, 3)`` uint8 RGB ndarray.

        Must be safe to call without holding ``self._lock`` from the caller;
        it acquires the lock internally.
        """
        with self._lock:
            sensors = self._nt.sensors
            cam = sensors.SensorTiledCamera(model=self._model)
            cam.utils.create_default_light(enable_shadows=False)
            rays = cam.utils.compute_pinhole_camera_rays(w, h, math.radians(50.0))
            color = cam.utils.create_color_image_output(w, h, 1)
            eye = (0.6, 0.6, 0.5)
            target = (0.0, 0.0, 0.15)
            q = self._look_at_quat(eye, target)
            wp = self._wp
            cam_tf = wp.array([[wp.transformf(wp.vec3f(*eye), wp.quatf(*q))]], dtype=wp.transformf)
            self._model.bvh_refit_shapes(self._state_0)
            cam.update(
                self._state_0,
                cam_tf,
                rays,
                color_image=color,
                clear_data=sensors.SensorTiledCamera.GRAY_CLEAR_DATA,
            )
            rgba = cam.utils.to_rgba_from_color(color).numpy()
        frame = rgba[0, 0] if rgba.ndim == 5 else rgba[0]
        return np.ascontiguousarray(frame[..., :3])

    # Introspection

    def describe(self) -> dict[str, Any]:
        """Return a discovery surface describing this backend's capabilities.

        Returns:
            Dict with the backend name, active solver, available solvers,
            device, and current robot / object counts.
        """
        device = str(self._wp.get_device(self.device)) if self.device else str(self._wp.get_device())
        return {
            "backend": "newton",
            "solver": self._solver_name,
            "available_solvers": sorted(solver_registry()),
            "device": device,
            "robots": self.list_robots(),
            "objects": list(self._world.objects) if self._world else [],
            "timestep": self._world.timestep if self._world else self.default_timestep,
        }

    def cleanup(self, policy_stop_timeout: float | None = None) -> None:
        """Release resources (alias for :meth:`destroy`)."""
        self.destroy()

    def __enter__(self) -> NewtonSimEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

    # Internal helpers

    def _look_at_quat(self, eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0, 0, 1)) -> tuple:
        """Build an (x, y, z, w) quaternion orienting a camera at ``eye`` to look at ``target``.

        Uses an OpenGL-style camera frame (camera looks down its local -Z).

        When ``up`` is parallel to the view axis (e.g. a top-down camera
        directly above its target with the default world-up), an alternate up
        vector is chosen so the basis stays well-defined instead of producing a
        NaN quaternion.

        Args:
            eye: Camera position.
            target: Point the camera looks at.
            up: World up vector.

        Returns:
            Quaternion as ``(x, y, z, w)`` for ``warp.quatf``.

        Raises:
            ValueError: If ``eye`` and ``target`` coincide, leaving the view
                direction undefined.
        """
        e = np.asarray(eye, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        u = np.asarray(up, dtype=np.float64)
        f = t - e
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-9:
            raise ValueError(f"look_at: eye and target coincide ({eye!r}); camera direction is undefined.")
        f /= f_norm
        z = -f
        x = np.cross(u, z)
        # When ``up`` is (near-)parallel to the view axis -- e.g. a top-down
        # camera placed directly above its target with the default world-up --
        # ``cross(up, z)`` collapses to ~0 and normalising it would yield a
        # NaN quaternion (a silently garbage camera pose). Fall back to an
        # alternate up vector that is guaranteed non-parallel to ``z``.
        if np.linalg.norm(x) < 1e-6:
            alt_up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(alt_up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        r = np.stack([x, y, z], axis=1)
        return self._quat_from_matrix(r)

    @staticmethod
    def _quat_from_matrix(r: np.ndarray) -> tuple:
        """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
        trace = np.trace(r)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        return (x, y, z, w)

    def _advance(self, n_steps: int) -> None:
        """Run ``n_steps`` control steps (each ``substeps`` solver steps).

        Must be called with ``self._lock`` held.
        """
        assert self._world is not None
        if self._solver is None:
            self._world.step_count += max(1, n_steps)
            self._world.sim_time += self._world.timestep * max(1, n_steps)
            return
        dt = self._world.timestep / self.substeps
        for _ in range(max(1, n_steps)):
            for _ in range(self.substeps):
                self._state_0.clear_forces()
                self._solver.step(self._state_0, self._state_1, self._control, None, dt)
                self._state_0, self._state_1 = self._state_1, self._state_0
            self._world.sim_time += self._world.timestep
            self._world.step_count += 1

    def _write_targets(self) -> None:
        """Push pending position targets into the Newton control buffer.

        Must be called with ``self._lock`` held.
        """
        if self._control is None or self._control.joint_target_q is None:
            return
        tgt = self._control.joint_target_q.numpy()
        for (robot_name, jname), value in self._targets.items():
            idx = self._joint_coord_index.get((robot_name, jname))
            if idx is not None and idx < len(tgt):
                tgt[idx] = value
        self._control.joint_target_q = self._wp.array(tgt, dtype=self._wp.float32, device=self._model.device)

    def _rebuild(self) -> None:
        """(Re)build the Newton model from the current world state.

        Newton finalises an immutable model from a builder, so every scene
        mutation triggers a full rebuild. Joint targets that still reference
        existing joints are preserved. Must be called with ``self._lock`` held.
        """
        assert self._world is not None
        nt, wp = self._nt, self._wp
        builder = nt.ModelBuilder()
        solver_cls = resolve_solver_class(self._solver_name)
        if hasattr(solver_cls, "register_custom_attributes"):
            solver_cls.register_custom_attributes(builder)

        # Track coordinate index of each robot's joints in the global joint_q.
        self._joint_coord_index = {}
        self._robot_joint_map = {}

        for robot_name, robot in self._world.robots.items():
            coord_before = builder.joint_coord_count
            label_before = len(builder.joint_label)
            xform = wp.transform(
                wp.vec3(*robot.position),
                self._wxyz_to_wp_quat(robot.orientation),
            )
            builder.add_mjcf(str(robot.urdf_path), xform=xform, collapse_fixed_joints=True)
            new_labels = builder.joint_label[label_before:]
            short_names: list[str] = []
            for offset, label in enumerate(new_labels):
                short = _short_joint_name(label)
                short_names.append(short)
                self._joint_coord_index[(robot_name, short)] = coord_before + offset
            self._robot_joint_map[robot_name] = short_names

        for obj in self._world.objects.values():
            self._add_object_to_builder(builder, obj)

        if self._world.ground_plane:
            builder.add_ground_plane()

        self._model = builder.finalize(device=self.device)
        # Rigid-body solvers (notably SolverMuJoCo) require at least one joint.
        # An empty world (ground plane only) has none, so defer solver creation
        # until a robot is added; stepping is a no-op until then.
        self._solver = solver_cls(self._model) if self._model.joint_dof_count > 0 else None
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        nt.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)
        self._joint_order = [name for names in self._robot_joint_map.values() for name in names]

        # Re-apply targets that still reference live joints.
        self._targets = {k: v for k, v in self._targets.items() if k in self._joint_coord_index}
        self._write_targets()

    def _add_object_to_builder(self, builder: Any, obj: SimObject) -> None:
        """Add one :class:`SimObject` primitive to a Newton builder."""
        wp = self._wp
        xform = wp.transform(wp.vec3(*obj.position), self._wxyz_to_wp_quat(obj.orientation))
        if obj.is_static or obj.mass <= 0:
            body = -1
        else:
            body = builder.add_body(xform=xform, mass=obj.mass)
        shape_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()) if body >= 0 else xform
        color = tuple(obj.color[:3])
        size = obj.size
        if obj.shape == "box":
            hx, hy, hz = (size + [0.05, 0.05, 0.05])[:3]
            builder.add_shape_box(body, xform=shape_xform, hx=hx, hy=hy, hz=hz, color=color)
        elif obj.shape == "sphere":
            builder.add_shape_sphere(body, xform=shape_xform, radius=size[0], color=color)
        elif obj.shape == "capsule":
            radius = size[0]
            half_height = size[1] if len(size) > 1 else size[0]
            builder.add_shape_capsule(body, xform=shape_xform, radius=radius, half_height=half_height, color=color)
        elif obj.shape == "cylinder":
            radius = size[0]
            half_height = size[1] if len(size) > 1 else size[0]
            builder.add_shape_cylinder(body, xform=shape_xform, radius=radius, half_height=half_height, color=color)

    def _wxyz_to_wp_quat(self, wxyz: list[float]) -> Any:
        """Convert a wxyz quaternion (SimRobot convention) to a warp xyzw quatf."""
        w, x, y, z = wxyz
        return self._wp.quat(x, y, z, w)
