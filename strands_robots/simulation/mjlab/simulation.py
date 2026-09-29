"""``MjlabEngine`` - the mjlab (MuJoCo-Warp) backend of the SimEngine ABC.

Design (see ``docs/reference/backends/mjlab.md`` and the lane STUDY):

* **One world, N copies.** mjlab replicates the scene over ``num_envs`` worlds
  and steps them in one Warp kernel launch. World 0 is what the single-world
  ``SimEngine`` contract sees; :meth:`get_observation_batch` and
  :meth:`send_action_batch` reach all worlds.
* **Lazy build.** Robots and objects are collected as mjlab ``EntityCfg`` and
  the scene is compiled on the first call that needs physics (``step``,
  ``reset``, ``get_observation``...). Adding or removing an entity after the
  build recompiles the scene, the same rebuild-on-mutation rule the Newton
  backend follows. Joint positions are carried across rebuilds by name.
* **Plain MJCF in, plain MJCF actuators.** Robots load from the same asset
  paths as the MuJoCo backend (``resolve_model_path``), keep their XML
  ``<actuator>`` blocks (``XmlActuatorCfg``) and ``send_action`` writes
  ``ctrl`` directly, so a ctrl trajectory replayed on both backends produces
  the same joint trajectory (parity tests in ``tests/simulation/mjlab``).
* **Option parity.** mjlab drops a child spec's ``<option>`` when attaching
  it to the scene, so the first robot's ``timestep``/``integrator``/``solver``
  options are copied into ``MujocoCfg`` unless the caller pinned them.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.assets import resolve_model_path, resolve_robot_name
from strands_robots.simulation.base import (
    SimEngine,
    own_keyword_names,
    reject_misspelled_kwargs,
    reject_setup_kwargs,
)
from strands_robots.simulation.mjlab.randomization import MjlabRandomizationMixin
from strands_robots.simulation.mjlab.recording import MjlabRecordingMixin
from strands_robots.utils import coerce_pose_vector, entity_name_error, positive_count_error

if TYPE_CHECKING:
    import mujoco
    import torch

logger = logging.getLogger(__name__)

_DEFAULT_TIMESTEP = 0.002
_STEPS_PER_BATCH = 1000
# mjlab (MuJoCo-Warp) supports euler + implicitfast: RK4/implicit fall back to implicitfast.
_INTEGRATORS = {0: "euler", 1: "implicitfast", 2: "implicitfast", 3: "implicitfast"}
_SOLVERS = {0: "pgs", 1: "cg", 2: "newton"}
_CONE = {0: "pyramidal", 1: "elliptic"}
_SHAPE_GEOM = {"box": "BOX", "sphere": "SPHERE", "cylinder": "CYLINDER", "capsule": "CAPSULE"}


def ensure_mjlab() -> Any:
    """Import mjlab or raise an ``ImportError`` that names the extra."""
    try:
        import mjlab  # noqa: F401
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The mjlab backend needs mjlab, mujoco-warp and warp-lang: "
            "pip install 'strands-robots[sim-mjlab]' (pins mujoco~=3.11, torch>=2.14)."
        ) from exc
    import mjlab

    return mjlab


@dataclass
class _RobotSpec:
    """One robot the caller added; enough to rebuild its ``EntityCfg``."""

    name: str
    path: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    keyframe: str | int | None
    joint_names: list[str] = field(default_factory=list)
    actuator_names: list[str] = field(default_factory=list)
    free_base: bool = False
    home_qpos: dict[str, float] = field(default_factory=dict)


@dataclass
class _ObjectSpec:
    name: str
    shape: str
    size: tuple[float, ...]
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    mass: float
    color: tuple[float, float, float, float]
    static: bool


class MjlabEngine(MjlabRandomizationMixin, MjlabRecordingMixin, SimEngine):
    """GPU-vectorized MuJoCo backend built on mjlab / MuJoCo-Warp.

    Args:
        num_envs: Number of world copies stepped together (``>= 1``).
        device: Torch/Warp device (``"cuda:0"`` default when CUDA is available,
            else ``"cpu"``).
        default_timestep: Physics timestep used by ``create_world`` when the
            caller passes none. The first robot's MJCF ``<option timestep>``
            wins over this default (MuJoCo-backend parity) unless
            ``create_world(timestep=...)`` was called explicitly.
        env_spacing: Distance in metres between world origins (visual only;
            worlds never collide).
        nconmax: Contact buffer size per world for MuJoCo-Warp (``None`` lets
            mjlab size it).
        njmax: Constraint buffer size per world (``None`` = mjlab default).
        use_cuda_graph: Capture ``step`` into a CUDA graph after the first
            step (faster at large ``num_envs``; disabled on CPU).
        default_width: Render width for :meth:`render` when unspecified.
        default_height: Render height for :meth:`render` when unspecified.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str | None = None,
        default_timestep: float = _DEFAULT_TIMESTEP,
        env_spacing: float = 2.0,
        nconmax: int | None = None,
        njmax: int | None = None,
        use_cuda_graph: bool = False,
        default_width: int = 640,
        default_height: int = 480,
        **kwargs: Any,
    ) -> None:
        reject_setup_kwargs(kwargs)
        reject_misspelled_kwargs(kwargs, own_keyword_names(MjlabEngine), owner="MjlabEngine")
        super().__init__()
        self._lock = threading.RLock()
        for value, param in (
            (num_envs, "num_envs"),
            (default_width, "default_width"),
            (default_height, "default_height"),
        ):
            err = positive_count_error(value, param, "MjlabEngine")
            if err:
                raise ValueError(err)
        if not isinstance(default_timestep, (int, float)) or not default_timestep > 0:
            raise ValueError(f"MjlabEngine: default_timestep must be a positive number, got {default_timestep!r}")
        self.num_envs = int(num_envs)
        self.env_spacing = float(env_spacing)
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self._default_timestep = float(default_timestep)
        self._nconmax = nconmax
        self._njmax = njmax
        self._use_cuda_graph = bool(use_cuda_graph)

        ensure_mjlab()
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        # World options (create_world) - None = "take it from the first robot".
        self._timestep: float | None = None
        self._timestep_pinned = False
        self._gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        self._ground_plane = True
        self._world_created = False
        self._integrator: str | None = None
        self._solver: str | None = None
        self._iterations: int | None = None
        self._ls_iterations: int | None = None
        self._cone: str | None = None

        self._robots: dict[str, _RobotSpec] = {}
        self._objects: dict[str, _ObjectSpec] = {}
        self._cameras: dict[str, dict[str, Any]] = {}
        self._recording_state_dict: dict[str, Any] = {}

        # Built state (None until the first physics call).
        self._scene: Any = None
        self._sim: Any = None
        self._dirty = True
        self._step_count = 0
        self._pending_ctrl: dict[str, np.ndarray] = {}
        self._renderer: Any = None
        self._render_data: Any = None
        self._build_seconds = 0.0
        self._dr_applied: dict[str, Any] | None = None
        # Last on purpose: SimEngine.__del__ only runs cleanup on engines that finished __init__.
        self._init_complete = True

    # ------------------------------------------------------------------ world

    def create_world(
        self,
        timestep: float | None = None,
        gravity: Sequence[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure the shared world: timestep, gravity, flat ground (terrains are not supported yet)."""
        with self._lock:
            if timestep is not None:
                if not isinstance(timestep, (int, float)) or not timestep > 0:
                    return {"status": "error", "content": [{"text": f"timestep must be positive, got {timestep!r}"}]}
                self._timestep = float(timestep)
                self._timestep_pinned = True
            if gravity is not None:
                g, err = coerce_pose_vector("create_world", "gravity", gravity, 3)
                if err:
                    return {"status": "error", "content": [{"text": err}]}
                assert g is not None
                self._gravity = (g[0], g[1], g[2])
            if terrain not in (None, "flat", "plane"):
                return {
                    "status": "error",
                    "content": [{"text": f"MjlabEngine: terrain {terrain!r} is not supported yet (flat plane only)."}],
                }
            self._ground_plane = bool(ground_plane)
            self._world_created = True
            self._dirty = True
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"World configured: num_envs={self.num_envs} device={self.device} "
                            f"dt={self._timestep if self._timestep is not None else 'robot default'} "
                            f"g={list(self._gravity)} ground={self._ground_plane}"
                        )
                    }
                ],
            }

    def destroy(self) -> dict[str, Any]:
        """Tear down the compiled scene and forget every robot, object and camera."""
        with self._lock:
            self._teardown_built()
            self._robots.clear()
            self._objects.clear()
            self._cameras.clear()
            self._pending_ctrl.clear()
            self._recording_state_dict = {}
            self._world_created = False
            self._dirty = True
            return {"status": "success", "content": [{"text": "World destroyed"}]}

    def _teardown_built(self) -> None:
        self._scene = None
        self._sim = None
        self._renderer = None
        self._render_data = None
        self._step_count = 0

    # ---------------------------------------------------------------- building

    def _mjcf_option(self, path: str) -> dict[str, Any]:
        import mujoco

        spec = mujoco.MjSpec.from_file(path)
        opt = spec.option
        return {
            "timestep": float(opt.timestep),
            "integrator": _INTEGRATORS.get(int(opt.integrator), "EULER"),
            "solver": _SOLVERS.get(int(opt.solver), "NEWTON"),
            "iterations": int(opt.iterations),
            "ls_iterations": int(opt.ls_iterations),
            "cone": _CONE.get(int(opt.cone), "PYRAMIDAL"),
        }

    def _robot_entity_cfg(self, spec: _RobotSpec) -> Any:
        from mjlab.actuator import XmlActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        path = spec.path

        def spec_fn(_path: str = path) -> Any:
            import mujoco

            return mujoco.MjSpec.from_file(_path)

        # mjlab resolves patterns first-match, so explicit joints go before the catch-all.
        joint_pos: dict[str, float] = {f"^{re.escape(k)}$": float(v) for k, v in spec.home_qpos.items()}
        joint_pos[".*"] = 0.0
        actuators = (XmlActuatorCfg(target_names_expr=(".*",)),) if spec.actuator_names else ()
        return EntityCfg(
            spec_fn=spec_fn,
            init_state=EntityCfg.InitialStateCfg(pos=spec.position, rot=spec.orientation, joint_pos=joint_pos),
            articulation=EntityArticulationInfoCfg(actuators=actuators) if actuators else None,
        )

    def _object_entity_cfg(self, obj: _ObjectSpec) -> Any:
        from mjlab.entity import EntityCfg

        def spec_fn(_obj: _ObjectSpec = obj) -> Any:
            import mujoco

            s = mujoco.MjSpec()
            body = s.worldbody.add_body(name=_obj.name)
            if not _obj.static:
                body.add_freejoint()
            geom = body.add_geom(
                name=f"{_obj.name}_geom",
                type=getattr(mujoco.mjtGeom, f"mjGEOM_{_SHAPE_GEOM[_obj.shape]}"),
                size=list(_obj.size) + [0.0] * (3 - len(_obj.size)),
                rgba=list(_obj.color),
            )
            if _obj.mass > 0 and not _obj.static:
                geom.mass = _obj.mass
            return s

        return EntityCfg(
            spec_fn=spec_fn,
            init_state=EntityCfg.InitialStateCfg(pos=obj.position, rot=obj.orientation),
        )

    def _build(self) -> None:
        """Compile the scene + simulation from the collected entity specs."""
        import torch
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

        t0 = time.perf_counter()
        saved = self._snapshot_joint_positions() if self._sim is not None else {}
        self._teardown_built()

        entities: dict[str, Any] = {}
        for name, spec in self._robots.items():
            entities[name] = self._robot_entity_cfg(spec)
        for name, obj in self._objects.items():
            entities[name] = self._object_entity_cfg(obj)

        scene_cfg = SceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing, entities=entities)
        scene = Scene(scene_cfg, device=self.device)
        # mjlab's scene.xml has no floor (tasks add terrain cfgs); a plane geom
        # on the worldbody gives the MuJoCo backend's default ground.
        model = self._compile_with_plane(scene) if self._ground_plane else scene.compile()

        first = next(iter(self._robots.values()), None)
        opt = self._mjcf_option(first.path) if first is not None else {}
        timestep = self._timestep if self._timestep is not None else opt.get("timestep", self._default_timestep)
        self._timestep = float(timestep)
        mj_cfg = MujocoCfg(
            timestep=self._timestep,
            gravity=self._gravity,
            integrator=self._integrator or opt.get("integrator", "euler"),
            solver=self._solver or opt.get("solver", "newton"),
            iterations=self._iterations or opt.get("iterations", 100),
            ls_iterations=self._ls_iterations or opt.get("ls_iterations", 50),
            cone=self._cone or opt.get("cone", "pyramidal"),
        )
        sim_kwargs: dict[str, Any] = {"mujoco": mj_cfg}
        if self._nconmax is not None:
            sim_kwargs["nconmax"] = int(self._nconmax)
        if self._njmax is not None:
            sim_kwargs["njmax"] = int(self._njmax)
        elif any(r.free_base for r in self._robots.values()):
            # mjlab's default njmax (from the compiled model's own nefc) is sized
            # for a robot standing on its feet; a floating-base robot that falls
            # or a fallen humanoid on the plane needs a few times more rows
            # (measured: G1 collapsing hits 70 rows, default overflowed at ~48).
            sim_kwargs["njmax"] = 8 * max(1, model.nv)
        sim = Simulation(num_envs=self.num_envs, cfg=SimulationCfg(**sim_kwargs), model=model, device=self.device)
        scene.initialize(sim.mj_model, sim.model, sim.data)
        scene.reset()
        sim.forward()
        scene.update(self._timestep)

        self._scene = scene
        self._sim = sim
        self._dirty = False
        self._build_seconds = time.perf_counter() - t0

        for name, spec in self._robots.items():
            ent = scene[name]
            spec.joint_names = self._contract_joint_names(spec, ent)
            spec.actuator_names = list(ent.actuator_names)
            spec.free_base = not ent.is_fixed_base
        # A fresh scene spawns in its declared pose right away, as the MuJoCo
        # backend does at add_robot(keyframe=...) time; a rebuild after a
        # mutation carries the live joint positions over instead.
        if saved:
            self._restore_joint_positions(saved)
        else:
            self._write_spawn_state()
        if self._use_cuda_graph and self.device.startswith("cuda"):
            try:
                sim.create_graph()
            except Exception as exc:  # pragma: no cover - depends on driver
                logger.warning("MjlabEngine: CUDA graph capture failed (%s); stepping eagerly", exc)
        logger.info(
            "MjlabEngine built: num_envs=%d nq=%d nu=%d robots=%s objects=%s in %.1fs",
            self.num_envs,
            sim.mj_model.nq,
            sim.mj_model.nu,
            list(self._robots),
            list(self._objects),
            self._build_seconds,
        )
        del torch

    def _compile_with_plane(self, scene: Any) -> Any:
        """Recompile the scene spec with a ground plane geom on the worldbody."""
        import mujoco

        spec = scene.spec
        if any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
            return scene.compile()
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            rgba=[0.55, 0.55, 0.55, 1.0],
        )
        spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
        return scene.compile()

    def _contract_joint_names(self, spec: _RobotSpec, ent: Any) -> list[str]:
        """Joint list in MJCF order, free joint first, as the MuJoCo backend reports it."""
        import mujoco

        model = self._sim.mj_model
        names: list[str] = []
        prefix = f"{spec.name}/"
        for j in range(model.njnt):
            full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if full and full.startswith(prefix):
                names.append(full[len(prefix) :])
        return names

    def _ensure_built(self) -> None:
        if self._sim is None or self._dirty:
            if not self._robots and not self._objects:
                raise RuntimeError("MjlabEngine: add a robot or object before stepping (empty scene).")
            self._build()

    # --------------------------------------------------------- joint snapshots

    def _snapshot_joint_positions(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        model = self._sim.mj_model
        qpos = self._sim.data.qpos.cpu().numpy()
        import mujoco

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr, width = model.jnt_qposadr[j], _qpos_width(model.jnt_type[j])
            out[name] = qpos[:, adr : adr + width].copy()
        return out

    def _restore_joint_positions(self, saved: Mapping[str, np.ndarray]) -> None:
        import mujoco
        import torch

        model = self._sim.mj_model
        qpos = self._sim.data.qpos.cpu().numpy()
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in saved and saved[name].shape[0] == qpos.shape[0]:
                adr, width = model.jnt_qposadr[j], _qpos_width(model.jnt_type[j])
                qpos[:, adr : adr + width] = saved[name][:, :width]
        self._sim.data.qpos[:] = torch.as_tensor(qpos, device=self.device)
        self._sim.forward()
        self._scene.update(self._timestep or self._default_timestep)

    def _write_spawn_state(self) -> None:
        """Every world to its declared spawn pose (default root state + env origins,
        default joint pose), zero velocity, zero ctrl, then forward."""
        import torch

        origins = self._scene.env_origins
        for ent in self._scene.entities.values():
            d = ent.data
            if not ent.is_fixed_base and d.default_root_state is not None:
                root = d.default_root_state.clone()
                root[:, 0:3] += origins
                ent.write_root_state_to_sim(root)
            if ent.is_articulated and d.default_joint_pos is not None and d.default_joint_pos.shape[-1] > 0:
                ent.write_joint_state_to_sim(d.default_joint_pos.clone(), torch.zeros_like(d.default_joint_pos))
        self._sim.data.qvel[:] = 0.0
        self._sim.data.ctrl[:] = 0.0
        self._sim.forward()
        self._scene.update(self._timestep or self._default_timestep)

    # ----------------------------------------------------------------- stepping

    def reset(self) -> dict[str, Any]:
        """Return every world to its spawn pose with zero velocity and zero ctrl.

        An open recording episode is saved first (episode boundary), as on the
        other backends; a failed flush leaves the worlds untouched.
        """
        flush_note = ""
        if (flush := self._flush_open_episode_before_reset()) is not None:
            if flush.get("status") != "success":
                return flush
            flush_note = flush["content"][0]["text"] + " "
        with self._lock:
            self._ensure_built()
            # mjlab's Scene.reset() only clears actuator state; restoring the
            # spawn pose is the job of the RL env's reset events. Do it here
            # the same way (default root state + env origins, default joints).
            self._scene.reset()
            self._write_spawn_state()
            self._pending_ctrl.clear()
            self._step_count = 0
            return {
                "status": "success",
                "content": [{"text": f"{flush_note}Reset {self.num_envs} world(s) to initial state"}],
            }

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance all worlds ``n_steps`` physics steps (the lock is released every 1000 steps)."""
        err = positive_count_error(n_steps, "n_steps", "step")
        if err:
            return {"status": "error", "content": [{"text": err}]}
        remaining = int(n_steps)
        while remaining > 0:
            batch = min(remaining, _STEPS_PER_BATCH)
            with self._lock:
                self._ensure_built()
                dt = self._timestep or self._default_timestep
                for _ in range(batch):
                    self._sim.step()
                    self._scene.update(dt)
                self._step_count += batch
            remaining -= batch
        return {"status": "success", "content": [{"text": f"Stepped {n_steps} x {self.num_envs} worlds"}]}

    def get_state(self) -> dict[str, Any]:
        """One-screen summary: time, step count, num_envs, device, entity counts and model sizes."""
        with self._lock:
            built = self._sim is not None and not self._dirty
            dt = self._timestep if self._timestep is not None else self._default_timestep
            t = self._step_count * dt
            lines = [
                "Simulation State (mjlab)",
                f"t={t:.4f}s (step {self._step_count}) | num_envs={self.num_envs} | device={self.device}",
                f"dt={dt}s | g={list(self._gravity)} | built={built}",
                f"Robots: {len(self._robots)} | Objects: {len(self._objects)} | Cameras: {len(self._cameras)}",
            ]
            if built:
                m = self._sim.mj_model
                lines.append(f"Bodies: {m.nbody} | Joints: {m.njnt} | Actuators: {m.nu} | nq={m.nq}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def physics_timestep(self) -> float | None:  # method, like the ABC (base.py) and the MuJoCo backend
        """The physics timestep in seconds (robot MJCF default until ``create_world`` pins one)."""
        return self._timestep if self._timestep is not None else self._default_timestep

    # ------------------------------------------------------------------ robots

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: Sequence[float] | None = None,
        orientation: Sequence[float] | None = None,
        keyframe: str | int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a registry robot (or an explicit MJCF) to every world; the scene recompiles lazily."""
        err = entity_name_error("add_robot", "name", name)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        pos, err = coerce_pose_vector("add_robot", "position", position, 3)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        quat, err = coerce_pose_vector("add_robot", "orientation", orientation, 4)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        with self._lock:
            if name in self._robots or name in self._objects:
                return {"status": "error", "content": [{"text": f"Entity '{name}' already exists"}]}
            if urdf_path:
                path = str(Path(urdf_path).expanduser())
                if not Path(path).exists():
                    return {"status": "error", "content": [{"text": f"Model file not found: {path}"}]}
                if path.lower().endswith(".urdf"):
                    return {
                        "status": "error",
                        "content": [{"text": "MjlabEngine loads MJCF only; convert the URDF or use backend='newton'."}],
                    }
            else:
                try:
                    path = str(resolve_model_path(resolve_robot_name(name)))
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    return {
                        "status": "error",
                        "content": [{"text": f"Could not resolve a sim asset for '{name}': {exc}"}],
                    }
            spec = _RobotSpec(
                name=name,
                path=path,
                position=tuple(pos) if pos else (0.0, 0.0, 0.0),
                orientation=tuple(quat) if quat else (1.0, 0.0, 0.0, 0.0),
                keyframe=keyframe,
            )
            try:
                spec.home_qpos, spec.free_base, spec.actuator_names, spec.joint_names = _inspect_mjcf(path, keyframe)
                if spec.free_base and position is None:
                    spec.position = _keyframe_root_pos(path, keyframe) or spec.position
            except Exception as exc:
                return {"status": "error", "content": [{"text": f"Could not load MJCF for '{name}': {exc}"}]}
            self._robots[name] = spec
            self._dirty = True
            return {
                "status": "success",
                "content": [
                    {"text": f"Added robot '{name}' ({len(spec.actuator_names)} actuators) to {self.num_envs} world(s)"}
                ],
                "joint_names": list(spec.joint_names),
                "action_keys": list(spec.actuator_names),
            }

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Drop a robot; the scene recompiles on the next physics call."""
        with self._lock:
            if name not in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{name}' not found"}]}
            del self._robots[name]
            self._pending_ctrl.pop(name, None)
            self._dirty = True
            return {"status": "success", "content": [{"text": f"Removed robot '{name}'"}]}

    def list_robots(self) -> list[str]:
        """Names of the robots in the scene."""
        return list(self._robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Joint names in MJCF order, free joint first (the recording column order)."""
        spec = self._robots.get(robot_name)
        if spec is None:
            raise KeyError(f"Robot '{robot_name}' not found")
        return list(spec.joint_names)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        """Actuator names in MJCF order: the order a numeric ``send_action`` vector uses."""
        spec = self._robots.get(robot_name)
        if spec is None:
            raise KeyError(f"Robot '{robot_name}' not found")
        return list(spec.actuator_names)

    def actuator_ranges(self, robot_name: str) -> dict[str, tuple[float, float]]:
        """``{action_key: (lo, hi)}`` from the MJCF ctrlrange (unbounded when unlimited)."""
        with self._lock:
            self._ensure_built()
            model = self._sim.mj_model
            ids = self._actuator_ids(robot_name)
            out: dict[str, tuple[float, float]] = {}
            for key, aid in zip(self._robots[robot_name].actuator_names, ids, strict=True):
                lo, hi = model.actuator_ctrlrange[aid]
                out[key] = (float(lo), float(hi)) if model.actuator_ctrllimited[aid] else (-np.inf, np.inf)
            return out

    def _actuator_ids(self, robot_name: str) -> list[int]:
        import mujoco

        model = self._sim.mj_model
        ids = []
        for key in self._robots[robot_name].actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot_name}/{key}")
            if aid < 0:
                raise KeyError(f"actuator {robot_name}/{key} missing after build")
            ids.append(aid)
        return ids

    # ----------------------------------------------------------------- objects

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: Sequence[float] | None = None,
        orientation: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        color: Sequence[float] | None = None,
        mass: float = 0.1,
        is_static: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a primitive (box/sphere/cylinder/capsule) as a free or static body to every world.

        Parameter order is the one every backend shares (graded by
        ``tests/simulation/test_backend_shared_parameter_order.py``); ``static`` is
        accepted as the older spelling of ``is_static``.
        """
        if "static" in kwargs and is_static is None:
            is_static = bool(kwargs.pop("static"))
        static = bool(is_static)
        err = entity_name_error("add_object", "name", name)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        if shape not in _SHAPE_GEOM:
            return {
                "status": "error",
                "content": [{"text": f"shape must be one of {sorted(_SHAPE_GEOM)}, got {shape!r}"}],
            }
        pos, err = coerce_pose_vector("add_object", "position", position, 3)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        quat, err = coerce_pose_vector("add_object", "orientation", orientation, 4)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        size_t = tuple(float(s) for s in (size or (0.02, 0.02, 0.02)))
        rgba = tuple(float(c) for c in (color or (0.8, 0.2, 0.2, 1.0)))
        if len(rgba) == 3:
            rgba = (*rgba, 1.0)
        with self._lock:
            if name in self._robots or name in self._objects:
                return {"status": "error", "content": [{"text": f"Entity '{name}' already exists"}]}
            self._objects[name] = _ObjectSpec(
                name=name,
                shape=shape,
                size=size_t,
                position=tuple(pos) if pos else (0.0, 0.0, 0.0),
                orientation=tuple(quat) if quat else (1.0, 0.0, 0.0, 0.0),
                mass=float(mass),
                color=rgba,
                static=bool(static),
            )
            self._dirty = True
            return {"status": "success", "content": [{"text": f"Added {shape} '{name}'"}]}

    def remove_object(self, name: str) -> dict[str, Any]:
        """Drop an object; the scene recompiles on the next physics call."""
        with self._lock:
            if name not in self._objects:
                return {"status": "error", "content": [{"text": f"Object '{name}' not found"}]}
            del self._objects[name]
            self._dirty = True
            return {"status": "success", "content": [{"text": f"Removed object '{name}'"}]}

    def list_objects(self) -> list[str]:
        """Names of the objects in the scene."""
        return list(self._objects)

    # ------------------------------------------------------------ observations

    def _default_robot(self, robot_name: str | None) -> str:
        if robot_name is None:
            if len(self._robots) != 1:
                raise ValueError("robot_name is required when the scene has 0 or several robots")
            return next(iter(self._robots))
        if robot_name not in self._robots:
            raise KeyError(f"Robot '{robot_name}' not found")
        return robot_name

    def get_observation_batch(self, robot_name: str | None = None) -> dict[str, torch.Tensor]:
        """All worlds at once: ``{joint: (N,), joint.vel: (N,), base_*: (N, k)}`` tensors."""
        with self._lock:
            robot_name = self._default_robot(robot_name)
            self._ensure_built()
            ent = self._scene[robot_name]
            spec = self._robots[robot_name]
            jp, jv = ent.data.joint_pos, ent.data.joint_vel
            out: dict[str, Any] = {}
            for i, j in enumerate(ent.joint_names):
                out[j] = jp[:, i]
                out[f"{j}.vel"] = jv[:, i]
            if spec.free_base:
                # Same frames as the classic backend's free-joint block
                # (mujoco/rendering.py): lin vel WORLD, ang vel BODY (IMU gyro).
                out["base_pos"] = ent.data.root_link_pos_w
                out["base_quat"] = ent.data.root_link_quat_w
                out["base_lin_vel"] = ent.data.root_link_lin_vel_w
                out["base_ang_vel"] = ent.data.root_link_ang_vel_b
            return out

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """World-0 observation: ``joint``, ``joint.vel`` floats, ``base_*`` lists for a floating base, camera images."""
        batch = self.get_observation_batch(robot_name)
        out: dict[str, Any] = {}
        for k, v in batch.items():
            row = v[0].detach().cpu().numpy()
            out[k] = float(row) if row.ndim == 0 else [float(x) for x in row]
        if not skip_images:
            for cam in self._cameras:
                try:
                    out[cam] = self.render(cam)
                except Exception as exc:  # pragma: no cover - render backend specific
                    logger.debug("render %s failed: %s", cam, exc)
        return out

    def send_action_batch(
        self, action: torch.Tensor | np.ndarray, robot_name: str | None = None, n_substeps: int = 1
    ) -> dict[str, Any]:
        """Write an ``(N, nu)`` ctrl block (one row per world) and advance ``n_substeps`` steps."""
        import torch

        with self._lock:
            robot_name = self._default_robot(robot_name)
            self._ensure_built()
            ids = self._actuator_ids(robot_name)
            block = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            if block.shape != (self.num_envs, len(ids)):
                return {
                    "status": "error",
                    "content": [{"text": f"expected shape ({self.num_envs}, {len(ids)}), got {tuple(block.shape)}"}],
                }
            self._sim.data.ctrl[:, ids] = block
            dt = self._timestep or self._default_timestep
            for _ in range(int(n_substeps)):
                self._sim.step()
                self._scene.update(dt)
            self._step_count += int(n_substeps)
            return {
                "status": "success",
                "content": [{"text": f"ctrl written for {self.num_envs} worlds, advanced {n_substeps} step(s)"}],
            }

    def send_action(
        self,
        action: Mapping[str, float] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Write ``ctrl`` for one robot in every world, then advance ``n_substeps`` physics steps.

        Same contract as the MuJoCo backend: ``PolicyRunner`` calls this once
        per control step and never calls :meth:`step` itself, so the world
        must move here. A dict may name a subset of the action keys (the
        others keep their ctrl); a sequence must be complete, in
        :meth:`robot_action_keys` order.
        """
        err = positive_count_error(n_substeps, "n_substeps", "send_action")
        if err:
            return {"status": "error", "content": [{"text": err}]}
        with self._lock:
            robot_name = self._default_robot(robot_name)
            self._ensure_built()
            keys = self._robots[robot_name].actuator_names
            if isinstance(action, Mapping):
                unknown = [k for k in action if k not in keys]
                if unknown:
                    return {"status": "error", "content": [{"text": f"unknown action keys {unknown}; valid: {keys}"}]}
                vec = np.array([float(action.get(k, np.nan)) for k in keys], dtype=np.float32)
            else:
                vec = np.asarray(list(action), dtype=np.float32)
                if vec.shape != (len(keys),):
                    return {"status": "error", "content": [{"text": f"expected {len(keys)} values in order {keys}"}]}
            if not np.all(np.isfinite(vec[~np.isnan(vec)])):
                return {"status": "error", "content": [{"text": "action contains inf"}]}
            ids = self._actuator_ids(robot_name)
            import torch

            current = self._sim.data.ctrl[:, ids]
            new = torch.as_tensor(vec, device=self.device).expand(self.num_envs, -1).clone()
            mask = torch.isnan(new)
            new[mask] = current[mask]
            self._sim.data.ctrl[:, ids] = new
            dt = self._timestep or self._default_timestep
            for _ in range(int(n_substeps)):
                self._sim.step()
                self._scene.update(dt)
            self._step_count += int(n_substeps)
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Applied {int((~mask[0]).sum())} ctrl values to {self.num_envs} worlds, advanced {n_substeps} step(s)"
                    }
                ],
            }

    # ------------------------------------------------------------------ render

    def add_camera(
        self,
        name: str,
        position: Sequence[float] | None = None,
        target: Sequence[float] | None = None,
        fov: float = 60.0,
        width: int | None = None,
        height: int | None = None,
        parent_body: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Register a look-at camera used by :meth:`render` and image observations.

        ``fov`` is stored and applied to the world-0 renderer; ``parent_body`` (a
        camera mounted on a moving body) is not supported by the batched worlds
        and is refused rather than silently rendered from a fixed pose.
        """
        if parent_body is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_camera(parent_body=...) is not supported by the mjlab backend: the batched "
                            "worlds have no per-body camera; use the MuJoCo backend for mounted cameras."
                        )
                    }
                ],
            }
        err = entity_name_error("add_camera", "name", name)
        if err:
            return {"status": "error", "content": [{"text": err}]}
        with self._lock:
            self._cameras[name] = {
                "position": list(position) if position is not None else [1.0, -1.0, 0.8],
                "target": list(target) if target is not None else [0.0, 0.0, 0.2],
                "fov": float(fov),
                "width": int(width or self.default_width),
                "height": int(height or self.default_height),
            }
            return {"status": "success", "content": [{"text": f"Camera '{name}' added"}]}

    def render(
        self, camera_name: str | None = None, width: int | None = None, height: int | None = None, env_id: int = 0
    ) -> np.ndarray:
        """Offscreen RGB of one world through ``mujoco.Renderer`` (env-0 qpos copied from the GPU)."""
        import mujoco

        with self._lock:
            self._ensure_built()
            model = self._sim.mj_model
            cam_cfg = self._cameras.get(camera_name or "", None)
            w = int(width or (cam_cfg or {}).get("width", self.default_width))
            h = int(height or (cam_cfg or {}).get("height", self.default_height))
            if self._renderer is None or (self._renderer.width, self._renderer.height) != (w, h):
                self._renderer = mujoco.Renderer(model, height=h, width=w)
                self._render_data = mujoco.MjData(model)
            data = self._render_data
            qpos = self._sim.data.qpos[env_id].detach().cpu().numpy()
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam: Any
            if camera_name and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) >= 0:
                cam = camera_name
            else:
                cam = mujoco.MjvCamera()
                cfg = cam_cfg or {"position": [1.2, -1.2, 0.9], "target": [0.0, 0.0, 0.2]}
                p, t = np.array(cfg["position"], dtype=float), np.array(cfg["target"], dtype=float)
                d = p - t
                cam.lookat[:] = t
                cam.distance = float(np.linalg.norm(d))
                cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
                cam.elevation = float(-np.degrees(np.arcsin(d[2] / max(cam.distance, 1e-9))))
                if cam_cfg and "fov" in cam_cfg:
                    # Free cameras take the global vertical FOV; apply the registered one.
                    model.vis.global_.fovy = float(cam_cfg["fov"])
            self._renderer.update_scene(data, camera=cam)
            return self._renderer.render().copy()

    # ------------------------------------------------------------------- misc

    def describe(self) -> dict[str, Any]:
        """Backend descriptor with ``num_envs``, ``device``, build state and build time."""
        base = super().describe() if hasattr(super(), "describe") else {}
        base.update(
            {
                "backend": "mjlab",
                "num_envs": self.num_envs,
                "device": self.device,
                "built": self._sim is not None and not self._dirty,
                "build_seconds": round(self._build_seconds, 2),
            }
        )
        return base

    @property
    def mj_model(self) -> mujoco.MjModel | None:
        """The compiled ``mujoco.MjModel`` shared by every world (``None`` before the first build)."""
        return None if self._sim is None else self._sim.mj_model

    @property
    def sim(self) -> Any:
        """The underlying ``mjlab.sim.Simulation`` (``None`` before the first build)."""
        return self._sim

    @property
    def scene(self) -> Any:
        """The underlying ``mjlab.scene.Scene`` (``None`` before the first build)."""
        return self._scene

    def cleanup(self, policy_stop_timeout: float | None = None) -> None:
        """Stop a running policy and release the compiled scene."""
        try:
            stop = getattr(self, "stop_policy", None)
            if callable(stop):
                stop()
        finally:
            with self._lock:
                self._teardown_built()

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self._teardown_built()
        except Exception:
            pass


# ------------------------------------------------------------------ helpers


def _qpos_width(jnt_type: int) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[int(jnt_type)]


def _inspect_mjcf(path: str, keyframe: str | int | None) -> tuple[dict[str, float], bool, list[str], list[str]]:
    """Home joint positions (from the requested keyframe, else the zero pose), free-base flag, actuator and joint names."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    joints = [model.joint(j).name for j in range(model.njnt)]
    actuators = [model.actuator(a).name for a in range(model.nu)]
    free = model.njnt > 0 and int(model.jnt_type[0]) == 0
    home: dict[str, float] = {}
    key_id = _resolve_key(model, keyframe)
    if key_id is not None:
        qpos = model.key_qpos[key_id]
        for j in range(model.njnt):
            if int(model.jnt_type[j]) in (2, 3):
                home[model.joint(j).name] = float(qpos[model.jnt_qposadr[j]])
    return home, free, actuators, joints


def _keyframe_root_pos(path: str, keyframe: str | int | None) -> tuple[float, float, float] | None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    if model.njnt == 0 or int(model.jnt_type[0]) != 0:
        return None
    key_id = _resolve_key(model, keyframe)
    # keyframe=None: the zero configuration, whose root pose is the MJCF body
    # pos (qpos0), exactly where the MuJoCo backend spawns a free-base robot.
    q = model.key_qpos[key_id] if key_id is not None else model.qpos0
    return (float(q[0]), float(q[1]), float(q[2]))


def _resolve_key(model: Any, keyframe: str | int | None) -> int | None:
    """Keyframe id for ``add_robot(keyframe=...)``, or ``None`` for the zero pose.

    Mirrors the MuJoCo backend's contract: ``keyframe=None`` keeps the all-zero
    configuration even when the MJCF declares keyframes. (Finding F10: silently
    spawning from keyframe 0 made the stock ``g1.xml`` topple under a crouch hold
    that the classic backend survives from ``qpos0``; same model, same CPU
    MuJoCo, different start.)
    """
    if model.nkey == 0 or keyframe is None:
        return None
    if isinstance(keyframe, int):
        return keyframe if 0 <= keyframe < model.nkey else None
    import mujoco

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, str(keyframe))
    return kid if kid >= 0 else None
