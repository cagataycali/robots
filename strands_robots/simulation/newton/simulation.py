"""Newton GPU-native simulation backend — ``SimEngine`` implementation.

This module provides :class:`NewtonSimulation`, a GPU-accelerated physics
simulation backend built on NVIDIA Warp and Newton.  It implements the
:class:`~strands_robots.simulation.base.SimEngine` abstract interface.

Heavy dependencies (``warp``, ``newton``) are imported lazily on the
first call that requires them (e.g. ``create_world``), so importing
this module is fast and safe on machines without a GPU.

.. note::

   This is PR 1/7 of the Newton backend migration.  All ``SimEngine``
   methods are present but most raise ``NotImplementedError`` until the
   corresponding implementation PR lands.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.solvers import SOLVER_MAP

logger = logging.getLogger(__name__)

# Lazily imported — set by ``_ensure_deps()``
_warp = None
_newton = None


def _ensure_deps() -> tuple[Any, Any]:
    """Lazily import Warp and Newton, raising clear errors if missing."""
    global _warp, _newton  # noqa: PLW0603

    if _warp is not None and _newton is not None:
        return _warp, _newton

    try:
        import warp as wp

        _warp = wp
    except ImportError as exc:
        raise ImportError(
            "warp-lang is required for the Newton backend.  Install with: pip install 'strands-robots[newton]'"
        ) from exc

    try:
        import newton as nt

        _newton = nt
    except ImportError as exc:
        raise ImportError(
            "newton-sim is required for the Newton backend.  Install with: pip install 'strands-robots[newton]'"
        ) from exc

    return _warp, _newton


class NewtonSimulation(SimEngine):
    """GPU-native simulation backend built on NVIDIA Warp + Newton.

    Supports 4096+ parallel environments on a single GPU, 7 solver
    backends, differentiable simulation, and soft-body/cloth/MPM.

    Parameters
    ----------
    config : NewtonConfig | None
        Backend configuration.  Uses sensible defaults when ``None``.

    Examples
    --------
    >>> from strands_robots.simulation import create_simulation
    >>> sim = create_simulation("newton")
    >>> sim.create_world()
    >>> sim.add_robot("so100")
    >>> sim.step(100)
    >>> sim.destroy()
    """

    def __init__(self, config: NewtonConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = NewtonConfig(**kwargs) if kwargs else NewtonConfig()
        self._config = config

        # Warp/Newton modules — populated by _lazy_init()
        self._wp: Any = None
        self._nt: Any = None

        # Newton simulation objects
        self._builder: Any = None
        self._model: Any = None
        self._solver: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        self._renderer: Any = None

        # Tracking
        self._robots: dict[str, dict[str, Any]] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._sensors: dict[str, Any] = {}
        self._world_created: bool = False
        self._replicated: bool = False
        self._step_count: int = 0
        self._sim_time: float = 0.0

        logger.info(
            "NewtonSimulation created — solver=%s, device=%s, num_envs=%d",
            config.solver,
            config.device,
            config.num_envs,
        )

    @property
    def config(self) -> NewtonConfig:
        """Return the backend configuration (read-only)."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Import Warp/Newton on first use and initialise Warp."""
        if self._wp is not None:
            return
        wp, nt = _ensure_deps()
        self._wp = wp
        self._nt = nt
        try:
            wp.init()
            logger.info("Warp initialised on %r.", self._config.device)
        except Exception as exc:
            # GPU unavailable — fall back to CPU so users get a clear
            # error at step() time rather than a cryptic Warp crash.
            logger.warning(
                "Warp init on %r failed (%s); falling back to 'cpu'.",
                self._config.device,
                exc,
            )
            self._config.device = "cpu"

    def _ensure_world(self) -> None:
        """Raise if the world has not been created yet."""
        if not self._world_created:
            raise RuntimeError("World not created. Call create_world() first.")

    def _get_solver_class(self) -> Any:
        """Look up the Newton solver class for the configured solver."""
        class_name = SOLVER_MAP[self._config.solver]
        nt = self._nt
        # Newton 1.x organises solvers under newton.solvers or at top level
        solver_cls = getattr(getattr(nt, "solvers", nt), class_name, None)
        if solver_cls is None:
            raise RuntimeError(
                f"Solver {self._config.solver!r} ({class_name}) not found in the installed Newton version."
            )
        return solver_cls

    # ------------------------------------------------------------------
    # Required SimEngine methods
    # ------------------------------------------------------------------

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world.

        Initialises Warp/Newton lazily on first call, creates a
        ``ModelBuilder``, configures gravity and ground plane, and
        instantiates the selected solver.

        Parameters
        ----------
        timestep : float | None
            Override ``config.physics_dt`` for this world.
        gravity : list[float] | None
            Gravity vector ``[gx, gy, gz]``. Defaults to ``[0, -9.81, 0]``.
        ground_plane : bool
            Whether to add a ground plane at ``y=0``.

        Returns
        -------
        dict
            ``{"status": "created", "solver": ..., "device": ...}``.
        """
        # Stub — full implementation in PR 2/7
        self._lazy_init()
        self._world_created = True
        return {
            "status": "created",
            "solver": self._config.solver,
            "device": self._config.device,
            "num_envs": self._config.num_envs,
        }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources."""
        self._builder = None
        self._model = None
        self._solver = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._renderer = None
        self._robots.clear()
        self._objects.clear()
        self._sensors.clear()
        self._world_created = False
        self._replicated = False
        self._step_count = 0
        self._sim_time = 0.0
        logger.info("NewtonSimulation destroyed.")
        return {"status": "destroyed"}

    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        self._ensure_world()
        # Stub — full implementation in PR 2/7
        self._step_count = 0
        self._sim_time = 0.0
        return {"status": "reset", "step_count": 0, "sim_time": 0.0}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by *n_steps* physics steps.

        Parameters
        ----------
        n_steps : int
            Number of physics steps to take.

        Returns
        -------
        dict
            Step count and elapsed simulation time.
        """
        self._ensure_world()
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        # Stub — full implementation in PR 2/7
        self._step_count += n_steps
        self._sim_time += n_steps * self._config.physics_dt
        return {
            "status": "stepped",
            "n_steps": n_steps,
            "total_steps": self._step_count,
            "sim_time": self._sim_time,
        }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary."""
        return {
            "status": "ok",
            "world_created": self._world_created,
            "replicated": self._replicated,
            "step_count": self._step_count,
            "sim_time": self._sim_time,
            "num_envs": self._config.num_envs,
            "solver": self._config.solver,
            "device": self._config.device,
            "robots": list(self._robots.keys()),
            "objects": list(self._objects.keys()),
            "sensors": list(self._sensors.keys()),
        }

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Supports URDF/MJCF model files and procedural construction for
        known robots (so100, so101, koch).

        Parameters
        ----------
        name : str
            Unique robot identifier.
        urdf_path : str | None
            Path to URDF/MJCF model file.
        data_config : str | None
            Named data configuration for model resolution.
        position : list[float] | None
            Spawn position ``[x, y, z]``.
        orientation : list[float] | None
            Spawn orientation as quaternion ``[w, x, y, z]``.

        Returns
        -------
        dict
            Robot metadata including joint count.
        """
        self._ensure_world()
        if name in self._robots:
            raise ValueError(f"Robot {name!r} already exists in the simulation.")
        # Stub — full implementation in PR 2/7
        robot_info: dict[str, Any] = {
            "name": name,
            "urdf_path": urdf_path,
            "data_config": data_config,
            "position": position or [0.0, 0.0, 0.0],
            "orientation": orientation or [1.0, 0.0, 0.0, 0.0],
        }
        self._robots[name] = robot_info
        return {"status": "added", **robot_info}

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation."""
        if name not in self._robots:
            raise ValueError(f"Robot {name!r} not found.")
        del self._robots[name]
        return {"status": "removed", "name": name}

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
        """Add an object to the scene."""
        self._ensure_world()
        if name in self._objects:
            raise ValueError(f"Object {name!r} already exists in the simulation.")
        # Stub — full implementation in PR 3/7
        obj_info: dict[str, Any] = {
            "name": name,
            "shape": shape,
            "position": position or [0.0, 0.0, 0.0],
            "mass": mass,
            "is_static": is_static,
        }
        self._objects[name] = obj_info
        return {"status": "added", **obj_info}

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene."""
        if name not in self._objects:
            raise ValueError(f"Object {name!r} not found.")
        del self._objects[name]
        return {"status": "removed", "name": name}

    def get_observation(
        self,
        robot_name: str | None = None,
        camera_name: str | None = None,
    ) -> dict[str, Any]:
        """Get observation from simulation.

        Returns joint positions, velocities, and body transforms for
        the specified robot.  When ``camera_name`` is given, also
        includes an RGB image.

        Parameters
        ----------
        robot_name : str | None
            Robot to observe.  If ``None`` and exactly one robot is
            loaded, that robot is used.
        camera_name : str | None
            Camera to render (if any).

        Returns
        -------
        dict
            Observation containing ``joint_q``, ``joint_qd``,
            ``body_q`` (numpy arrays) and optionally ``image``.
        """
        self._ensure_world()
        # Stub — full implementation in PR 3/7
        return {"status": "stub", "robot_name": robot_name}

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> None:
        """Apply action to simulation.

        Parameters
        ----------
        action : dict
            Joint-name → target-position mapping, or a raw numpy
            array of target joint positions.
        robot_name : str | None
            Target robot.  If ``None`` and exactly one robot is loaded,
            that robot is used.
        n_substeps : int
            Number of substeps to apply per action.
        """
        self._ensure_world()
        # Stub — full implementation in PR 3/7

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Render a camera view.

        Returns
        -------
        dict
            ``{"image": np.ndarray (H, W, 3, uint8), "depth": ...}``
        """
        self._ensure_world()
        # Stub — full implementation in PR 4/7
        return {"status": "stub", "camera_name": camera_name}

    # ------------------------------------------------------------------
    # Optional SimEngine overrides
    # ------------------------------------------------------------------

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a scene from URDF/MJCF/USD file."""
        self._ensure_world()
        # Stub — full implementation in PR 2/7
        return {"status": "stub", "scene_path": scene_path}

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a policy loop in the simulation."""
        self._ensure_world()
        # Stub — full implementation in PR 3/7
        return {"status": "stub", "robot_name": robot_name}

    def randomize(self, **kwargs: Any) -> dict[str, Any]:
        """Apply domain randomization."""
        self._ensure_world()
        # Stub — full implementation in PR 5/7
        return {"status": "stub"}

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information from the collision pipeline."""
        self._ensure_world()
        # Stub — full implementation in PR 3/7
        return {"status": "stub", "contacts": []}

    def cleanup(self) -> None:
        """Release all resources."""
        if self._world_created:
            self.destroy()

    # ------------------------------------------------------------------
    # Newton-specific extensions
    # ------------------------------------------------------------------

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the simulation across multiple parallel environments.

        Must be called after ``add_robot()`` and before the first
        ``step()``.  Clones the physics state N times for GPU-parallel
        execution.

        Parameters
        ----------
        num_envs : int | None
            Number of environments.  Defaults to ``config.num_envs``.

        Returns
        -------
        dict
            Replication status and environment count.
        """
        self._ensure_world()
        n = num_envs or self._config.num_envs
        # Stub — full implementation in PR 5/7
        self._replicated = True
        return {"status": "replicated", "num_envs": n}

    def run_diffsim(
        self,
        num_steps: int,
        loss_fn: Any,
        optimize_params: str,
        lr: float = 0.02,
        iterations: int = 200,
    ) -> dict[str, Any]:
        """Run differentiable simulation optimisation loop.

        Parameters
        ----------
        num_steps : int
            Forward simulation steps per iteration.
        loss_fn : callable
            Function ``(states) → scalar loss`` to minimise.
        optimize_params : str
            Name of the parameter tensor to optimise.
        lr : float
            Learning rate for gradient descent.
        iterations : int
            Number of optimisation iterations.

        Returns
        -------
        dict
            Optimisation result including final loss.
        """
        self._ensure_world()
        if not self._config.enable_differentiable:
            raise RuntimeError("Differentiable simulation requires NewtonConfig(enable_differentiable=True).")
        # Stub — full implementation in PR 6/7
        return {"status": "stub"}

    def solve_ik(
        self,
        robot_name: str,
        target_position: list[float],
        target_orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Solve inverse kinematics for the specified robot.

        Parameters
        ----------
        robot_name : str
            Target robot.
        target_position : list[float]
            End-effector target position ``[x, y, z]``.
        target_orientation : list[float] | None
            End-effector target orientation as quaternion.

        Returns
        -------
        dict
            Solution containing ``joint_q`` numpy array.
        """
        self._ensure_world()
        # Stub — full implementation in PR 6/7
        return {"status": "stub", "robot_name": robot_name}

    def add_sensor(
        self,
        name: str,
        kind: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a sensor (contact, IMU, or tiled camera).

        Parameters
        ----------
        name : str
            Unique sensor identifier.
        kind : str
            Sensor type: ``"contact"``, ``"imu"``, or ``"camera"``.

        Returns
        -------
        dict
            Sensor metadata.
        """
        self._ensure_world()
        if name in self._sensors:
            raise ValueError(f"Sensor {name!r} already exists.")
        # Stub — full implementation in PR 4/7
        self._sensors[name] = {"name": name, "kind": kind, **kwargs}
        return {"status": "added", "name": name, "kind": kind}

    def read_sensor(self, name: str) -> dict[str, Any]:
        """Read the latest value from a sensor.

        Parameters
        ----------
        name : str
            Sensor identifier.

        Returns
        -------
        dict
            Sensor reading.
        """
        self._ensure_world()
        if name not in self._sensors:
            raise ValueError(f"Sensor {name!r} not found.")
        # Stub — full implementation in PR 4/7
        return {"status": "stub", "name": name}

    def enable_dual_solver(
        self,
        articulated: str = "featherstone",
        soft: str = "vbd",
    ) -> None:
        """Enable dual-solver mode: one for rigid bodies, one for soft.

        Parameters
        ----------
        articulated : str
            Solver for articulated rigid bodies.
        soft : str
            Solver for soft bodies / cloth.

        Raises
        ------
        ValueError
            If either solver name is not recognised.
        """
        if articulated not in SOLVER_MAP:
            raise ValueError(f"Unknown articulated solver {articulated!r}.")
        if soft not in SOLVER_MAP:
            raise ValueError(f"Unknown soft solver {soft!r}.")
        # Stub — full implementation in PR 5/7
        logger.info(
            "Dual-solver mode: articulated=%s, soft=%s",
            articulated,
            soft,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NewtonSimulation(solver={self._config.solver!r}, "
            f"device={self._config.device!r}, "
            f"num_envs={self._config.num_envs}, "
            f"world={self._world_created})"
        )
