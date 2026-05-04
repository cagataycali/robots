"""MuJoCo Simulation — AgentTool orchestrator composing physics/rendering/policy mixins."""

import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.model_registry import (
    list_available_models,
    resolve_model,
)
from strands_robots.simulation.model_registry import (
    register_urdf as _register_urdf,
)
from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimStatus, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.mujoco.mjcf_builder import MJCFBuilder
from strands_robots.simulation.mujoco.physics import PhysicsMixin
from strands_robots.simulation.mujoco.randomization import RandomizationMixin
from strands_robots.simulation.mujoco.recording import RecordingMixin
from strands_robots.simulation.mujoco.rendering import RenderingMixin
from strands_robots.simulation.mujoco.scene_ops import (
    eject_body_from_scene,
    inject_camera_into_scene,
    inject_object_into_scene,
    inject_robot_into_scene,
)
from strands_robots.simulation.policy_runner import CooperativeStop

if TYPE_CHECKING:
    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

_TOOL_SPEC_PATH = Path(__file__).parent / "tool_spec.json"


class Simulation(
    PhysicsMixin,
    RenderingMixin,
    RecordingMixin,
    RandomizationMixin,
    SimEngine,
    AgentTool,
):
    """Programmatic MuJoCo simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language → tool actions.

    **Stateful session.** One MuJoCo world per instance; actions form an
    implicit state machine starting with ``create_world``. Tools that mutate
    the scene (``add_robot``, ``remove_robot``, ``add_object``, ``remove_object``, ``move_object``, ``add_camera``, ``remove_camera``,
    ``load_scene``) are NOT safe to call while a policy is running via
    ``start_policy`` — stop it first. Call ``destroy()`` or ``cleanup()`` at
    session end to release the ThreadPoolExecutor, temp dirs, and MuJoCo
    resources.
    """

    def __init__(
        self,
        tool_name: str = "sim",
        default_timestep: float = 0.002,
        default_width: int = 640,
        default_height: int = 480,
        mesh: bool = True,
        peer_id: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        self._world: SimWorld | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{tool_name}_sim")
        self._policy_threads: dict[str, Future] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        self._viewer_handle = None
        self._viewer_thread = None

        # Thread-local renderer cache — MuJoCo Renderer uses thread-local GL
        # contexts (CGL on macOS, GLX on Linux). Sharing renderers across
        # threads causes SIGSEGV in cgl.free(). Each thread gets its own.
        self._renderer_tls = threading.local()
        self._renderer_model = None

        # Fail fast: verify MuJoCo is importable at construction time
        # so consumers catch missing-dependency errors immediately.
        self._mj = _ensure_mujoco()
        logger.info("🎮 Simulation tool '%s' initialized", tool_name)

    # Public Properties

    @property
    def mj_model(self):
        """Direct access to the MuJoCo model (mujoco.MjModel)."""
        return self._world._model if self._world else None

    @property
    def mj_data(self):
        """Direct access to the MuJoCo data (mujoco.MjData)."""
        return self._world._data if self._world else None

    # Robot-compatible interface

    def get_observation(self, robot_name: str | None = None) -> dict[str, Any]:
        """Get full observation for a robot: joint state + all attached cameras.

        See :meth:`SimEngine.get_observation` for the schema contract.
        """
        if self._world is None or self._world._model is None:
            return {}
        if robot_name is None:
            if not self._world.robots:
                return {}
            robot_name = next(iter(self._world.robots))
        if robot_name not in self._world.robots:
            return {}
        return self._get_sim_observation(robot_name)

    def send_action(self, action: dict[str, Any], robot_name: str | None = None, n_substeps: int = 1) -> None:
        """Apply action to simulation (Robot ABC compatible).

        Thread-safety: acquires self._lock around ctrl writes + mj_step,
        as documented in base.py's SimEngine contract. Concurrent calls
        from the agent's dispatch thread and a PolicyRunner worker are
        serialized here.
        """
        if self._world is None or self._world._model is None:
            return
        if robot_name is None:
            if not self._world.robots:
                return
            robot_name = next(iter(self._world.robots))
        if robot_name not in self._world.robots:
            return
        with self._lock:
            self._apply_sim_action(robot_name, action, n_substeps=n_substeps)

    # World Management

    def _cheap_robot_count(self) -> int:
        try:
            from strands_robots.registry import list_robots as _registry_list_robots

            return len(_registry_list_robots(mode="sim"))
        except ImportError:
            return 0

    def create_world(
        self, timestep: float | None = None, gravity: list[float] | None = None, ground_plane: bool = True
    ) -> dict[str, Any]:
        """Create a new simulation world."""
        # mujoco verified at __init__

        if self._world is not None and self._world._model is not None:
            return {
                "status": "error",
                "content": [{"text": "World already exists. Use action='destroy' first, or action='reset'."}],
            }

        if gravity is None:
            _gravity = [0.0, 0.0, -9.81]
        elif isinstance(gravity, (int, float)):
            _gravity = [0.0, 0.0, float(gravity)]
        else:
            _gravity = list(gravity)

        self._world = SimWorld(
            timestep=timestep or self.default_timestep,
            gravity=_gravity,
            ground_plane=ground_plane,
        )

        self._world.cameras["default"] = SimCamera(
            name="default",
            position=[1.5, 1.5, 1.2],
            target=[0.0, 0.0, 0.3],
            width=self.default_width,
            height=self.default_height,
        )

        self._compile_world()

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        "🌍 Simulation world created\n"
                        f"⚙️ Timestep: {self._world.timestep}s ({1 / self._world.timestep:.0f}Hz physics)\n"
                        f"🌐 Gravity: {self._world.gravity}\n"
                        f"📷 Default camera ready\n"
                        f"🤖 Robot models: {self._cheap_robot_count()} available\n"
                        "💡 Add robots: action='add_robot' (urdf_path or data_config)\n"
                        "💡 Add objects: action='add_object'\n"
                        "💡 List URDFs: action='list_urdfs'"
                    )
                }
            ],
        }

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a complete scene from MJCF XML or URDF file."""
        if err := self._require_no_running_policy("load_scene"):
            return err
        mj = self._mj

        if not os.path.exists(scene_path):
            return {"status": "error", "content": [{"text": f"Scene file not found: {scene_path}"}]}

        try:
            self._world = SimWorld()
            self._world._model = mj.MjModel.from_xml_path(str(scene_path))
            self._world._data = mj.MjData(self._world._model)
            self._world.status = SimStatus.IDLE

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🌍 Scene loaded from {os.path.basename(scene_path)}\n"
                            f"🦴 Bodies: {self._world._model.nbody}, 🔩 Joints: {self._world._model.njnt}, ⚡ Actuators: {self._world._model.nu}\n"
                            "💡 Use action='get_state' to inspect, action='step' to simulate"
                        )
                    }
                ],
            }
        except Exception as e:
            logger.error("Failed to load scene: %s", e)
            return {"status": "error", "content": [{"text": f"Failed to load scene: {e}"}]}

    def _compile_world(self):
        mj = self._mj
        xml = MJCFBuilder.build_objects_only(self._world)
        self._world._backend_state["xml"] = xml
        self._world._model = mj.MjModel.from_xml_string(xml)
        self._world._data = mj.MjData(self._world._model)
        self._world.status = SimStatus.IDLE

    def _recompile_world(self) -> dict[str, Any]:
        try:
            self._compile_world()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Recompile failed: {e}"}]}

    # Robot Management

    @staticmethod
    def _ensure_meshes(model_path: str, robot_name: str):
        """Check if mesh files referenced by a model XML exist; auto-download if missing."""
        model_dir = os.path.dirname(os.path.abspath(model_path))

        files_to_check = [model_path]
        try:
            with open(model_path) as _f:
                top_content = _f.read()
            for inc in re.findall(r'<include\s+file="([^"]+)"', top_content):
                inc_path = os.path.join(model_dir, inc)
                if os.path.exists(inc_path):
                    files_to_check.append(inc_path)
        except Exception:
            pass

        missing = False
        for xml_path in files_to_check:
            try:
                with open(xml_path) as _f:
                    content = _f.read()
            except Exception:
                continue

            mesh_files = re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content)
            if not mesh_files:
                continue

            meshdir_match = re.search(r'meshdir="([^"]*)"', content)
            meshdir = meshdir_match.group(1) if meshdir_match else ""
            xml_dir = os.path.dirname(os.path.abspath(xml_path))

            for mf in mesh_files:
                if not os.path.exists(os.path.join(xml_dir, meshdir, mf)):
                    missing = True
                    break
            if missing:
                break

        if not missing:
            return

        logger.info("Downloading mesh files for '%s' from MuJoCo Menagerie (first time only)...", robot_name)
        try:
            from strands_robots.assets import resolve_robot_name
            from strands_robots.assets.download import download_robots

            canonical = resolve_robot_name(robot_name)
            download_robots(names=[canonical], force=True)
        except (ImportError, FileNotFoundError, OSError) as e:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Auto-download failed for '{robot_name}': {e}. "
                            f"Install robot_descriptions: pip install strands-robots[sim-mujoco]"
                        )
                    }
                ],
            }

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation via XML round-trip composition.

        Instead of replacing the entire world model, this method merges the
        robot's bodies, actuators, assets, and sensors into the existing scene
        XML.  This preserves previously-created world state (gravity, objects,
        cameras, other robots).
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Use action='create_world' first."}]}
        if err := self._require_no_running_policy("add_robot"):
            return err
        if name in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' already exists."}]}

        # Resolution precedence (T22/T49):
        #   1. explicit `urdf_path` (anything on disk).
        #   2. `data_config` looked up in the model registry.
        #   3. DEPRECATED: `name` looked up in the registry (undocumented
        #      fallback kept for one release with a DeprecationWarning).
        # Pass `data_config` for new code; the `name`-as-registry-key path
        # will be removed.
        resolved_path = urdf_path
        if not resolved_path and data_config:
            resolved_path = resolve_model(data_config)
            if not resolved_path:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"No model found for '{data_config}'.\n💡 Use action='list_urdfs' to see available robots"
                        }
                    ],
                }
        elif not resolved_path and name:
            # T22: deprecated fallback — try registry by instance name.
            import warnings as _warnings
            resolved_path = resolve_model(name)
            if resolved_path:
                _warnings.warn(
                    f"add_robot: resolving model via instance name '{name}' is deprecated; "
                    "pass data_config='<registry-key>' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if not resolved_path:
            return {"status": "error", "content": [{"text": "Either urdf_path or data_config is required."}]}
        if not os.path.exists(resolved_path):
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        mj = self._mj

        robot = SimRobot(
            name=name,
            urdf_path=resolved_path,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            data_config=data_config,
            namespace=f"{name}/",
        )

        try:
            self._ensure_meshes(resolved_path, data_config or name)

            # Pre-scan the robot XML to discover joint/actuator names.
            # We load a temporary model just for introspection — this is NOT
            # used as the world model.
            tmp_model = mj.MjModel.from_xml_path(str(resolved_path))

            joint_names = []
            for i in range(tmp_model.njnt):
                jnt_name = mj.mj_id2name(tmp_model, mj.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    joint_names.append(jnt_name)
            robot.joint_names = joint_names

            # Discover cameras from robot model
            for i in range(tmp_model.ncam):
                cam_name = mj.mj_id2name(tmp_model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name and cam_name not in self._world.cameras:
                    self._world.cameras[cam_name] = SimCamera(
                        name=cam_name,
                        camera_id=i,
                        width=self.default_width,
                        height=self.default_height,
                    )

            # Register the robot BEFORE injection so _reload_scene_from_xml
            # can re-discover its joint/actuator IDs in the merged model.
            self._world.robots[name] = robot
            # Track robot base path for asset path resolution.
            if not self._world._backend_state.get("robot_base_xml"):
                self._world._backend_state["robot_base_xml"] = resolved_path

            # XML round-trip: merge robot into existing world
            ok = inject_robot_into_scene(self._world, robot, resolved_path)
            if not ok:
                del self._world.robots[name]
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject robot '{name}' into scene."}],
                }

            # Re-read joint/actuator IDs from the merged model (IDs shifted).
            # Names inside MuJoCo are namespaced (e.g. ``arm0/shoulder_pan``)
            # when multiple same-config robots are injected, so prefer the
            # namespaced lookup.
            model = self._world._model
            pfx = robot.namespace or ""
            robot.joint_ids = []
            robot.actuator_ids = []
            for jnt_name in robot.joint_names:
                jid = -1
                if pfx:
                    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
                if jid < 0:
                    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                if jid >= 0:
                    robot.joint_ids.append(jid)
            for i in range(model.nu):
                jnt_id = model.actuator_trnid[i, 0]
                if jnt_id in robot.joint_ids:
                    robot.actuator_ids.append(i)
            if not robot.actuator_ids and len(self._world.robots) == 1:
                # Fallback: single-robot scene — assign all actuators.
                for i in range(model.nu):
                    robot.actuator_ids.append(i)

            # T6: leave the freshly-added robot in a clean, deterministic
            # zero state (qpos=qvel=ctrl=0) rather than silently settling
            # under gravity for 100 steps. Callers that want a pre-settled
            # pose should call step()/reset() explicitly. This makes
            # `add_robot` -> `get_robot_state` observations meaningful for
            # learning pipelines that expect t=0 to be a canonical start.
            mj.mj_resetData(self._world._model, self._world._data)
            self._world.sim_time = 0.0
            self._world.step_count = 0
            mj.mj_forward(self._world._model, self._world._data)

            source = f"data_config='{data_config}'" if data_config else os.path.basename(resolved_path)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🤖 Robot '{name}' added to simulation\n"
                            f"📁 Source: {source} → {os.path.basename(resolved_path)}\n"
                            f"📍 Position: {robot.position}\n"
                            f"🔩 Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                            f"⚡ Actuators: {len(robot.actuator_ids)}\n"
                            f"📷 Cameras: {list(self._world.cameras.keys())}\n"
                            f"💡 Run policy: action='run_policy', robot_name='{name}'"
                        )
                    }
                ],
            }
        except Exception as e:
            # Clean up on failure
            self._world.robots.pop(name, None)
            logger.error("Failed to add robot '%s': %s", name, e)
            return {"status": "error", "content": [{"text": f"Failed to load: {e}"}]}

    def remove_robot(self, name: str) -> dict[str, Any]:
        if self._world is None or name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' not found."}]}
        # Guard: remove_robot races the cooperative-stop path if the robot has an active policy.
        if err := self._require_no_running_policy("remove_robot"):
            return err
        if name in self._policy_threads:
            self._world.robots[name].policy_running = False
            try:
                self._policy_threads[name].result(timeout=5.0)
            except Exception:
                pass
            del self._policy_threads[name]
        del self._world.robots[name]
        return {"status": "success", "content": [{"text": f"🗑️ Robot '{name}' removed."}]}

    def list_robots(self) -> list[str]:
        """Return ordered robot names (SimEngine ABC).

        For the user-facing agent-tool action (rich dict output) see
        :meth:`list_robots_info`, which the dispatcher aliases to the
        ``list_robots`` action string.
        """
        if self._world is None or not self._world.robots:
            return []
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Ordered joint names for ``robot_name`` (SimEngine ABC)."""
        if self._world is None or robot_name not in self._world.robots:
            return []
        return list(self._world.robots[robot_name].joint_names)

    def list_robots_info(self) -> dict[str, Any]:
        """Agent-tool action: pretty-printed robot listing.

        Separate from :meth:`list_robots` (which returns ``list[str]`` for
        the SimEngine ABC) because the dispatcher needs a dict-shaped
        response for user display.
        """
        if err := self._require_world():
            return err
        if not self._world.robots:
            return {"status": "success", "content": [{"text": "No robots. Use action='add_robot'."}]}

        lines = ["🤖 Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            status = "🟢 running" if robot.policy_running else "⚪ idle"
            lines.append(
                f"  • {name} ({os.path.basename(robot.urdf_path)})\n"
                f"    Position: {robot.position}, Joints: {len(robot.joint_names)}, "
                f"Config: {robot.data_config or 'direct'}, Status: {status}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def get_robot_state(self, robot_name: str) -> dict[str, Any]:
        """T23: canonical name parameter is ``robot_name``. The router
        accepts ``name`` as an alias (bidirectional) so legacy LLM calls
        keep working, but new tool specs should document only robot_name."""
        if err := self._require_world():
            return err
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        mj = self._mj
        robot = self._world.robots[robot_name]
        model, data = self._world._model, self._world._data

        # Namespace-aware joint lookup (see add_robot / _apply_sim_action).
        pfx = robot.namespace or ""
        state = {}
        for jnt_name in robot.joint_names:
            jnt_id = -1
            if pfx:
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jnt_id < 0:
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                state[jnt_name] = {
                    "position": float(data.qpos[model.jnt_qposadr[jnt_id]]),
                    "velocity": float(data.qvel[model.jnt_dofadr[jnt_id]]),
                }

        text = f"🤖 '{robot_name}' state (t={self._world.sim_time:.3f}s):\n"
        for jnt, vals in state.items():
            text += f"{jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"

        return {"status": "success", "content": [{"text": text}, {"json": {"state": state}}]}

    # Object Management

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool | None = None,
        mesh_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object to the simulation."""
        if err := self._require_world():
            return err
        if err := self._require_no_running_policy("add_object"):
            return err
        if name in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' exists."}]}

        # T29: planes are infinite and must be static.  Explicit
        # is_static=False for a plane is an error; None or True both
        # resolve to True. Non-plane shapes default to dynamic.
        if shape == "plane":
            if is_static is False:
                return {
                    "status": "error",
                    "content": [
                        {"text": "add_object: shape='plane' requires is_static=True (planes are infinite and cannot have dynamic mass)."}
                    ],
                }
            is_static = True
        elif is_static is None:
            is_static = False

        obj = SimObject(
            name=name,
            shape=shape,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            size=size or [0.05, 0.05, 0.05],
            color=color or [0.5, 0.5, 0.5, 1.0],
            mass=mass,
            mesh_path=mesh_path,
            is_static=is_static,
        )
        self._world.objects[name] = obj

        if self._world.robots:
            try:
                result = inject_object_into_scene(self._world, obj)
                if result:
                    return {
                        "status": "success",
                        "content": [{"text": f"📦 '{name}' spawned: {shape} at {obj.position}"}],
                    }
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"📦 '{name}' registered: {shape} at {obj.position}\n"
                                "⚠️ Robot scene loaded — object is tracked but not physically spawned."
                            )
                        }
                    ],
                }
            except (ValueError, RuntimeError) as e:
                # Clean up: object was added to world.objects before injection
                self._world.objects.pop(name, None)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject '{name}' into live scene: {e}"}],
                }

        recompile_result = self._recompile_world()
        if recompile_result["status"] == "error":
            del self._world.objects[name]
            return recompile_result

        return {
            "status": "success",
            "content": [
                {
                    "text": f"📦 '{name}' added: {shape} at {obj.position}, size={obj.size}, {'static' if is_static else f'{mass}kg'}"
                }
            ],
        }

    def remove_object(self, name: str) -> dict[str, Any]:
        if self._world is None or name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        if err := self._require_no_running_policy("remove_object"):
            return err
        del self._world.objects[name]
        if self._world.robots:
            eject_body_from_scene(self._world, name)
        else:
            self._recompile_world()
        return {"status": "success", "content": [{"text": f"🗑️ '{name}' removed."}]}

    def move_object(
        self, name: str, position: list[float] | None = None, orientation: list[float] | None = None
    ) -> dict[str, Any]:
        if err := self._require_world():
            return err
        if name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        # Guard: move_object writes qpos + calls mj_forward, racing a running policy.
        if err := self._require_no_running_policy("move_object"):
            return err

        mj = self._mj
        model, data = self._world._model, self._world._data

        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if jnt_id >= 0:
            qpos_addr = model.jnt_qposadr[jnt_id]
            if position:
                data.qpos[qpos_addr : qpos_addr + 3] = position
                self._world.objects[name].position = position
            if orientation:
                data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
                self._world.objects[name].orientation = orientation
            mj.mj_forward(model, data)

        return {"status": "success", "content": [{"text": f"📍 '{name}' moved to {position or 'same'}"}]}

    def list_objects(self) -> dict[str, Any]:
        if err := self._require_world():
            return err
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}

        lines = ["📦 Objects:\n"]
        for name, obj in self._world.objects.items():
            lines.append(f"  • {name}: {obj.shape} at {obj.position}, {'static' if obj.is_static else f'{obj.mass}kg'}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # Camera Management

    def add_camera(
        self,
        name: str,
        position: list[float] | None = None,
        target: list[float] | None = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, Any]:
        if err := self._require_world():
            return err
        if err := self._require_no_running_policy("add_camera"):
            return err

        # T2: validate position / target shape before we bake them into XML.
        pos = position or [1.0, 1.0, 1.0]
        tgt = target or [0.0, 0.0, 0.0]
        for _lbl, _vec in (("position", pos), ("target", tgt)):
            try:
                if len(_vec) != 3:
                    return {"status": "error", "content": [{"text": f"add_camera: '{_lbl}' must be 3 elements [x,y,z], got {len(_vec)}"}]}
            except TypeError:
                return {"status": "error", "content": [{"text": f"add_camera: '{_lbl}' must be a list of 3 numbers"}]}
        # Degenerate orientation: position == target means no well-defined look direction.
        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):
            return {"status": "error", "content": [{"text": f"add_camera: 'position' and 'target' are identical ({pos}); camera has no look direction."}]}

        # T30/T41: reject duplicate camera names.  Previously a second
        # add_camera(name=existing) silently overwrote the registry entry but
        # left the XML's <camera> unchanged, so the old pose stuck around for
        # rendering.  Explicit error avoids the surprise.
        if name in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"add_camera: camera '{name}' already exists. Remove it first."}],
            }

        cam = SimCamera(
            name=name,
            position=pos,
            target=tgt,
            fov=fov,
            width=width,
            height=height,
        )
        self._world.cameras[name] = cam

        if self._world.robots and self._world._model is not None:
            try:
                inject_camera_into_scene(self._world, cam)
            except (ValueError, RuntimeError) as e:
                # Clean up: camera was added to world.cameras before injection
                self._world.cameras.pop(name, None)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject camera '{name}' into live scene: {e}"}],
                }
        else:
            self._recompile_world()

        return {"status": "success", "content": [{"text": f"📷 Camera '{name}' added at {cam.position}"}]}

    def remove_camera(self, name: str) -> dict[str, Any]:
        if self._world is None or name not in self._world.cameras:
            return {"status": "error", "content": [{"text": f"Camera '{name}' not found."}]}
        if err := self._require_no_running_policy("remove_camera"):
            return err
        del self._world.cameras[name]
        return {"status": "success", "content": [{"text": f"🗑️ Camera '{name}' removed."}]}

    # Simulation Control

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        if err := self._require_world():
            return err
        # T9: reject negative, accept zero as no-op
        if not isinstance(n_steps, int):
            try:
                n_steps = int(n_steps)
            except (TypeError, ValueError):
                return {"status": "error", "content": [{"text": f"step: n_steps must be an integer, got {type(n_steps).__name__}"}]}
        if n_steps < 0:
            return {"status": "error", "content": [{"text": f"step: n_steps must be >= 0, got {n_steps}"}]}
        if n_steps == 0:
            return {
                "status": "success",
                "content": [
                    {"text": f"⏩ +0 steps (no-op) | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}
                ],
            }
        mj = self._mj
        with self._lock:
            for _ in range(n_steps):
                mj.mj_step(self._world._model, self._world._data)
            self._world.sim_time = self._world._data.time
            self._world.step_count += n_steps
        return {
            "status": "success",
            "content": [
                {"text": f"⏩ +{n_steps} steps | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}
            ],
        }

    def reset(self) -> dict[str, Any]:
        if err := self._require_world():
            return err
        # T5: reset during a running policy races mj_step -> SEGFAULT risk
        if err := self._require_no_running_policy("reset"):
            return err
        mj = self._mj
        with self._lock:
            mj.mj_resetData(self._world._model, self._world._data)
            self._world.sim_time = 0.0
            self._world.step_count = 0
            # Flip policy_running flag inside the lock so a racing worker
            # thread cannot slip in one more mj_step between reset and flag
            # flip.
            for r in self._world.robots.values():
                r.policy_running = False
                r.policy_steps = 0
        return {"status": "success", "content": [{"text": "🔄 Reset to initial state."}]}

    def get_state(self) -> dict[str, Any]:
        if err := self._require_world():
            return err
        lines = [
            "🌍 Simulation State",
            f"🕐 t={self._world.sim_time:.4f}s (step {self._world.step_count})",
            f"⚙️ dt={self._world.timestep}s | 🌐 g={self._world.gravity}",
            f"🤖 Robots: {len(self._world.robots)} | 📦 Objects: {len(self._world.objects)} | 📷 Cameras: {len(self._world.cameras)}",
        ]
        if self._world._model:
            lines.append(
                f"🦴 Bodies: {self._world._model.nbody} | 🔩 Joints: {self._world._model.njnt} | ⚡ Actuators: {self._world._model.nu}"
            )
        if self._world._backend_state.get("recording", False):
            lines.append(f"🔴 Recording: {len(self._world._backend_state['trajectory'])} steps")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def destroy(self) -> dict[str, Any]:
        if self._world is None:
            return {"status": "success", "content": [{"text": "No world to destroy."}]}
        for r in self._world.robots.values():
            r.policy_running = False
        self._close_viewer()
        self._close_main_thread_renderers()
        self._world = None
        return {"status": "success", "content": [{"text": "🗑️ World destroyed."}]}

    def _close_main_thread_renderers(self) -> None:
        """T4: Close any renderers this thread owns and drop the TLS cache.

        Only safe for the main thread because ``mujoco.Renderer`` binds a
        CGL/GLX context to the thread that created it; closing from another
        thread can SIGSEGV in ``cgl.free()``. Worker threads drop their
        renderers via ``threading.Thread`` teardown.
        """
        tls = getattr(self, "_renderer_tls", None)
        if tls is None:
            return
        renderers = getattr(tls, "renderers", None)
        if renderers:
            for r in list(renderers.values()):
                try:
                    r.close()
                except Exception:
                    pass
            renderers.clear()
        # Forget the model marker so the next _get_renderer() rebuilds fresh.
        if hasattr(tls, "model"):
            tls.model = None

    def set_gravity(self, gravity: list[float] | float | int) -> dict[str, Any]:
        if err := self._require_world():
            return err
        # T5: set_gravity during a running policy races the worker thread
        if err := self._require_no_running_policy("set_gravity"):
            return err
        # T38: validate length/dtype before numpy broadcast
        if isinstance(gravity, (int, float)):
            gravity = [0.0, 0.0, float(gravity)]
        try:
            if len(gravity) != 3:
                return {"status": "error", "content": [{"text": f"set_gravity: 'gravity' must be a 3-element list [x,y,z], got {len(gravity)}"}]}
            gravity = [float(g) for g in gravity]
        except (TypeError, ValueError) as e:
            return {"status": "error", "content": [{"text": f"set_gravity: 'gravity' must be a 3-element list of numbers ({e})"}]}
        with self._lock:
            self._world._model.opt.gravity[:] = gravity
            self._world.gravity = gravity
        return {"status": "success", "content": [{"text": f"🌐 Gravity: {gravity}"}]}

    def set_timestep(self, timestep: float) -> dict[str, Any]:
        if err := self._require_world():
            return err
        # T5
        if err := self._require_no_running_policy("set_timestep"):
            return err
        # T8: reject non-positive; warn on huge values
        try:
            timestep = float(timestep)
        except (TypeError, ValueError):
            return {"status": "error", "content": [{"text": f"set_timestep: must be a positive number, got {timestep!r}"}]}
        if timestep <= 0:
            return {"status": "error", "content": [{"text": f"set_timestep: must be > 0, got {timestep}"}]}
        warn = ""
        if timestep > 0.1:
            warn = f" ⚠️ unusually large timestep (>{0.1}s); physics may be unstable"
        with self._lock:
            self._world._model.opt.timestep = timestep
            self._world.timestep = timestep
        return {"status": "success", "content": [{"text": f"⏱️ Timestep: {timestep}s ({1 / timestep:.0f}Hz){warn}"}]}

    # Viewer

    def open_viewer(self) -> dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "No simulation to view."}]}
        from strands_robots.simulation.mujoco.backend import _mujoco_viewer

        if _mujoco_viewer is None:
            return {"status": "error", "content": [{"text": "mujoco.viewer not available."}]}
        if self._viewer_handle is not None:
            return {"status": "success", "content": [{"text": "👁️ Viewer already open."}]}
        try:
            self._viewer_handle = _mujoco_viewer.launch_passive(self._world._model, self._world._data)
            return {"status": "success", "content": [{"text": "👁️ Interactive viewer opened."}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Viewer failed: {e}"}]}

    def _close_viewer(self) -> None:
        if self._viewer_handle is not None:
            try:
                self._viewer_handle.close()
            except Exception:
                pass
            self._viewer_handle = None

    def close_viewer(self) -> dict[str, Any]:
        self._close_viewer()
        return {"status": "success", "content": [{"text": "👁️ Viewer closed."}]}

    # URDF Registry

    def list_urdfs(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": list_available_models()}]}

    def register_urdf(self, data_config: str, urdf_path: str) -> dict[str, Any]:
        """T35: validate urdf_path before handing it to the registry.

        The router (T1) already rejects missing required params, so the
        no-args case produces a friendly 'requires parameter ...' message
        without hitting this body.
        """
        if not urdf_path:
            return {
                "status": "error",
                "content": [{"text": "register_urdf: 'urdf_path' must be a non-empty string."}],
            }
        p = Path(urdf_path)
        if not p.exists():
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: file not found: {urdf_path}"}],
            }
        if not p.is_file():
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: not a file: {urdf_path}"}],
            }
        try:
            # Smoke-check readability — mj.MjModel.from_xml_path will surface a
            # better error later, but permission issues are worth catching now.
            with p.open("rb"):
                pass
        except OSError as e:
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: cannot read {urdf_path}: {e}"}],
            }

        _register_urdf(data_config, urdf_path)
        resolved = resolve_model(data_config)
        return {
            "status": "success",
            "content": [{"text": f"📋 Registered '{data_config}' → {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"}],
        }

    # Introspection

    def get_features(self, robot_name: str | None = None) -> dict[str, Any]:
        """Describe the simulation's joints / actuators / cameras / robots.

        T33: If ``robot_name`` is given, the joint / actuator / camera listings
        are restricted to that robot (its namespaced MuJoCo names).  The
        ``robots`` map is also filtered to just that entry.
        """
        if err := self._require_world():
            return err

        mj = self._mj
        model = self._world._model

        # All-model name pools
        all_joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        all_joint_names = [n for n in all_joint_names if n]
        all_actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        all_actuator_names = [n for n in all_actuator_names if n]
        all_camera_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        all_camera_names = [n for n in all_camera_names if n]

        if robot_name is not None:
            if robot_name not in self._world.robots:
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
            robot = self._world.robots[robot_name]
            ns = (getattr(robot, "namespace", "") or "").rstrip("/")
            prefix = f"{ns}/" if ns else ""

            def _scoped(pool: list[str]) -> list[str]:
                if not prefix:
                    # Single-robot scene with no namespace: return the robot's own
                    # joints/actuators from the robot model rather than the pool.
                    return pool
                return [n for n in pool if n.startswith(prefix)]

            joint_names = robot.joint_names or _scoped(all_joint_names)
            actuator_names = _scoped(all_actuator_names)
            camera_names = _scoped(all_camera_names)

            robots_info = {
                robot_name: {
                    "joint_names": robot.joint_names,
                    "n_joints": len(robot.joint_names),
                    "n_actuators": len(robot.actuator_ids),
                    "data_config": robot.data_config,
                    "source": os.path.basename(robot.urdf_path),
                }
            }
        else:
            joint_names = all_joint_names
            actuator_names = all_actuator_names
            camera_names = all_camera_names

            robots_info = {}
            for rname, robot in self._world.robots.items():
                robots_info[rname] = {
                    "joint_names": robot.joint_names,
                    "n_joints": len(robot.joint_names),
                    "n_actuators": len(robot.actuator_ids),
                    "data_config": robot.data_config,
                    "source": os.path.basename(robot.urdf_path),
                }

        features = {
            "n_bodies": model.nbody,
            "n_joints": model.njnt,
            "n_actuators": model.nu,
            "n_cameras": model.ncam,
            "timestep": model.opt.timestep,
            "joint_names": joint_names,
            "actuator_names": actuator_names,
            "camera_names": camera_names,
            "robots": robots_info,
        }

        lines = [
            "🔍 Simulation Features",
            f"🦴 Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"⚡ Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
            f"📷 Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
            f"⏱️ Timestep: {model.opt.timestep}s ({1 / model.opt.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(
                f"🤖 {rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})"
            )

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}],
        }

    # AgentTool Interface

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "simulation"

    def _require_world(self) -> dict[str, Any] | None:
        """T14: Return unified 'no world' error or None if world is live.

        Replaces scattered ``"No simulation."`` / ``"No world."`` strings. Every
        action that touches ``self._world`` / ``self._world._model`` /
        ``self._world._data`` should call this first.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "No world. Call create_world (or load_scene) first."
                        )
                    }
                ],
            }
        return None

    def _require_no_running_policy(self, action_name: str) -> dict[str, Any] | None:
        """Return an error dict if a policy is running, else None.

        Scene mutations (add_robot, remove_robot, add_object, remove_object, move_object, add_camera, remove_camera,
        load_scene) swap model/data pointers via XML round-trip. A concurrent
        PolicyRunner worker calling mj_step on stale pointers is undefined
        behaviour. Hard-fail so the agent learns to stop the policy first.
        """
        has_running = any(not f.done() for f in self._policy_threads.values())
        if has_running:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Cannot '{action_name}' while a policy is running. Stop it first: action='stop_policy'."
                        )
                    }
                ],
            }
        return None

    @property
    def tool_spec(self) -> ToolSpec:
        with open(_TOOL_SPEC_PATH) as f:
            schema = json.load(f)
        return {
            "name": self.tool_name_str,
            "description": (
                "Programmatic MuJoCo simulation environment (stateful session). "
                "One world per instance; actions form an implicit state machine starting with "
                "create_world. Scene mutations (add_robot, remove_robot, add_object, remove_object, move_object, add_camera, remove_camera, "
                "load_scene) are blocked while a policy is running — stop it first. "
                "Create worlds, add robots from URDF "
                "(direct path or auto-resolve from data_config name), add objects, run VLA policies, "
                "render cameras, record trajectories, domain randomize. "
                "Same Policy ABC as real robot control — sim ↔ real with zero code changes. "
                "Actions: create_world, load_scene, reset, get_state, destroy, "
                "add_robot, remove_robot, list_robots, get_robot_state, "
                "add_object, remove_object, move_object, list_objects, "
                "add_camera, remove_camera, "
                "run_policy, start_policy, stop_policy, "
                "render, render_depth, render_all, get_contacts, "
                "step, set_gravity, set_timestep, "
                "randomize, "
                "start_recording, stop_recording, get_recording_status, start_cameras_recording, stop_cameras_recording, get_cameras_recording_status, "
                "open_viewer, close_viewer, "
                "list_urdfs, register_urdf, get_features. "
                "Call destroy() at session end to release resources."
            ),
            "inputSchema": {"json": schema},
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            result = self._dispatch_action(input_data.get("action", ""), input_data)
            yield ToolResultEvent(dict(toolUseId=tool_use_id, **result))  # type: ignore[typeddict-item]
        except Exception as e:
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"Sim error: {e}"}],
                }
            )

    # Policy orchestration overrides (MuJoCo-specific wiring)

    def start_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: "Policy | None" = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Start policy execution on a background thread (non-blocking).

        MuJoCo override: reuses the ThreadPoolExecutor owned by
        ``Simulation`` so agent tools can kick off long-running policies
        without blocking the event loop. Only one policy per robot at a
        time (MuJoCo model/data are not thread-safe for concurrent writes).

        T25: accepts ``n_steps`` (primary) or legacy ``max_steps`` as an
        alternate horizon specification; run_policy converts to duration.
        """
        if err := self._require_world():
            return err
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        existing = self._policy_threads.get(robot_name)
        if existing is not None and not existing.done():
            return {
                "status": "error",
                "content": [{"text": f"Policy already running on '{robot_name}'. Stop it first."}],
            }

        future = self._executor.submit(
            self.run_policy,
            robot_name,
            policy_provider=policy_provider,
            policy_config=policy_config,
            instruction=instruction,
            duration=duration,
            control_frequency=control_frequency,
            action_horizon=action_horizon,
            fast_mode=fast_mode,
            video=video,
            policy_object=policy_object,
            n_steps=n_steps,
            max_steps=max_steps,
        )
        self._policy_threads[robot_name] = future

        return {
            "status": "success",
            "content": [{"text": f"🚀 Policy started on '{robot_name}' (async)"}],
        }

    def _make_run_policy_hook(self, robot_name: str, instruction: str):
        """MuJoCo override: recording + policy_running flag + lock.

        Returns an ``on_frame(step, obs, action)`` closure that:
        * flips ``robot.policy_running`` so ``stop_policy`` can interrupt,
        * appends to ``_backend_state["trajectory"]`` when recording,
        * forwards frames to the LeRobot ``dataset_recorder`` if attached,
        * raises ``PolicyStopped`` when the user calls ``stop_policy``.
        """
        import numpy as np

        from strands_robots.simulation.models import TrajectoryStep

        world = self._world
        if world is None or robot_name not in world.robots:
            return None

        robot = world.robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0

        lock = self._lock

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            # Cooperative cancellation: stop_policy flips this flag.
            if not robot.policy_running:
                raise CooperativeStop(f"Policy stopped on '{robot_name}'")

            robot.policy_steps = step + 1

            with lock:
                if world._backend_state.get("recording", False):
                    world._backend_state["trajectory"].append(
                        TrajectoryStep(
                            timestamp=time.time(),
                            sim_time=world.sim_time,
                            robot_name=robot_name,
                            observation={k: v for k, v in observation.items() if not isinstance(v, np.ndarray)},
                            action=action,
                            instruction=instruction,
                        )
                    )
                    rec = world._backend_state.get("dataset_recorder")
                    if rec is not None:
                        rec.add_frame(observation=observation, action=action, task=instruction)

        return _hook

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: "Policy | None" = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """MuJoCo ``run_policy`` override: pre-flight world check + graceful stop.

        Delegates to :meth:`SimEngine.run_policy` but clears the MuJoCo
        ``policy_running`` flag in a ``finally`` clause and swallows
        ``_PolicyStopped`` (which the ``on_frame`` hook raises on user
        cancellation) into a normal "policy stopped" result.

        T25: forwards ``n_steps`` / ``max_steps`` to the base so LLM callers
        can specify horizon in steps rather than wall-clock seconds.
        """
        if err := self._require_world():
            return err

        try:
            return super().run_policy(
                robot_name,
                policy_provider=policy_provider,
                policy_config=policy_config,
                instruction=instruction,
                duration=duration,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=video,
                policy_object=policy_object,
                n_steps=n_steps,
                max_steps=max_steps,
            )
        finally:
            if self._world is not None and robot_name in self._world.robots:
                self._world.robots[robot_name].policy_running = False

    # Action name aliases (tool-action -> method-name)
    _ACTION_ALIASES = {
        "list_robots": "list_robots_info",
    }

    # Input field name -> method parameter name (syntactic sugar for the LLM)
    _FIELD_ALIASES = {
        "checkpoint_name": "name",
        "torque_vec": "torque",
    }

    # Params the router passes through but not every method declares.
    # These are used for cross-cutting concerns (e.g. video on run_policy)
    # and must not be reported as "unknown" by the router.
    _ROUTER_PASSTHROUGH = {"action"}

    # Vector params with expected length (for dimension validation before
    # numpy/MuJoCo sees them). Length 3 = xyz unless noted.
    _VECTOR_PARAM_LENGTHS: dict[str, int] = {
        "position": 3,
        "target": 3,
        "origin": 3,
        "force": 3,
        "torque": 3,
        "torque_vec": 3,
        "gravity": 3,
        "direction": 3,
        "point": 3,
        "orientation": 4,  # quaternion (w,x,y,z)
        "color": 4,  # rgba
    }

    def _validate_and_build_kwargs(
        self, action: str, method_name: str, sig: inspect.Signature, remapped: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """T1: Validate input against method signature; return (kwargs, error_result).

        Exactly one of the tuple elements is non-None.
        """
        # Strip self + VAR_POSITIONAL (*args) + VAR_KEYWORD (**kwargs) for signature
        # introspection; **kwargs methods accept arbitrary inputs, so we skip the
        # unknown-key check for them.
        named_params = {
            n: p
            for n, p in sig.parameters.items()
            if n != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        method_has_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        method_param_names = set(named_params)
        accepted_field_names = (
            method_param_names | set(self._FIELD_ALIASES.keys()) | self._ROUTER_PASSTHROUGH
        )

        # run_policy folds flat video keys into a structured `video` dict; those
        # flat keys are legitimate at the router boundary even though run_policy
        # itself takes `video=`.
        if action == "run_policy":
            accepted_field_names |= {"output_path", "fps", "camera_name"}

        # name/robot_name are aliased in both directions in the legacy router;
        # allow either here so we don't flag the alias as unknown.
        if "name" in method_param_names:
            accepted_field_names.add("robot_name")
        if "robot_name" in method_param_names:
            accepted_field_names.add("name")

        # 1) Unknown kwargs (skipped for **kwargs methods which legitimately passthrough)
        unknown = [] if method_has_var_keyword else [k for k in remapped if k not in accepted_field_names]
        if unknown:
            valid_sorted = sorted(method_param_names - {"action"})
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown parameter '{unknown[0]}' for action '{action}'. "
                            f"Valid: {valid_sorted}"
                        )
                    }
                ],
            }

        # 2) Vector dimension validation (applies before method runs)
        for vparam, expected_len in self._VECTOR_PARAM_LENGTHS.items():
            if vparam not in remapped:
                continue
            val = remapped[vparam]
            if val is None:
                continue
            if not hasattr(val, "__len__"):
                return None, {
                    "status": "error",
                    "content": [
                        {"text": f"Parameter '{vparam}' must be a list of {expected_len} numbers."}
                    ],
                }
            if len(val) != expected_len:
                return None, {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Parameter '{vparam}' must be a list of {expected_len} numbers, "
                                f"got {len(val)}."
                            )
                        }
                    ],
                }
            for i, component in enumerate(val):
                if not isinstance(component, (int, float)) or isinstance(component, bool):
                    return None, {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Parameter '{vparam}'[{i}] must be numeric, "
                                    f"got {type(component).__name__}."
                                )
                            }
                        ],
                    }

        # 3) Build kwargs + check required params
        kwargs: dict[str, Any] = {}
        for param_name, param in named_params.items():
            if param_name == "name" and "name" not in remapped and "robot_name" in remapped:
                kwargs["name"] = remapped["robot_name"]
            elif param_name == "robot_name" and "robot_name" not in remapped and "name" in remapped:
                kwargs["robot_name"] = remapped["name"]
            elif param_name in remapped:
                kwargs[param_name] = remapped[param_name]
            elif param.default is inspect.Parameter.empty:
                return None, {
                    "status": "error",
                    "content": [
                        {"text": f"Action '{action}' requires parameter '{param_name}'."}
                    ],
                }

        return kwargs, None

    def _dispatch_action(self, action: str, d: dict[str, Any]) -> dict[str, Any]:
        """Route action to the matching method with full input validation.

        Validation layer (T1):
          * unknown top-level params are rejected with a friendly message,
          * missing required params produce a "requires parameter X" error
            (no raw Python ``TypeError``),
          * vector params have length + numeric dtype checked before the
            value reaches numpy / MuJoCo.

        Policy-provider kwargs are nested under ``policy_config`` (never
        top-level) so the dispatcher stays backend-agnostic.
        """
        method_name = self._ACTION_ALIASES.get(action, action)
        method = getattr(self, method_name, None)

        if method is None or action.startswith("_"):
            return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

        cache = getattr(self, "_sig_cache", None)
        if cache is None:
            self._sig_cache = cache = {}
        if method_name not in cache:
            cache[method_name] = inspect.signature(method)
        sig = cache[method_name]

        # Field-alias rewriting (before validation so the validator sees
        # canonical names).
        remapped = {k: v for k, v in d.items() if k != "action"}
        for field_key, param_key in self._FIELD_ALIASES.items():
            if field_key in remapped and param_key not in remapped:
                remapped[param_key] = remapped.pop(field_key)

        # Fold flat video keys into `video` dict for run_policy/start_policy.
        if action in ("run_policy", "start_policy") and "video" not in remapped:
            _video_flat: dict[str, Any] = {}
            if "output_path" in remapped:
                _video_flat["path"] = remapped.pop("output_path")
            if "fps" in remapped:
                _video_flat["fps"] = remapped.pop("fps")
            # camera_name is shared with render(); only treat as video camera
            # when paired with an output path.
            if _video_flat.get("path") and "camera_name" in remapped:
                _video_flat["camera"] = remapped.pop("camera_name")
            if _video_flat.get("path"):
                remapped["video"] = _video_flat

        kwargs, err = self._validate_and_build_kwargs(action, method_name, sig, remapped)
        if err is not None:
            return err
        assert kwargs is not None
        return method(**kwargs)

    def stop_policy(self, robot_name: str = "") -> dict[str, Any]:
        """Stop a running policy on the given robot (cooperative cancellation).

        Counterpart to :meth:`start_policy`. Flips the robot's
        ``policy_running`` flag; the background loop in
        :meth:`_run_policy_loop` sees it and raises :class:`PolicyStopped`
        which is caught cleanly inside :meth:`start_policy`.

        T16: idempotent — if the robot exists but no policy is running, we
        still return success with 'Was not running' so callers can call
        stop_policy unconditionally. The only error case is an unknown
        robot_name.

        T24: empty robot_name returns a clear error instead of a silent
        match against the first robot.
        """
        if not robot_name:
            return {
                "status": "error",
                "content": [{"text": "stop_policy requires 'robot_name'."}],
            }
        if self._world is None or robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
        robot = self._world.robots[robot_name]
        was_running = robot.policy_running
        robot.policy_running = False
        msg = f"Stopped on '{robot_name}'" if was_running else f"Was not running on '{robot_name}'"
        return {"status": "success", "content": [{"text": msg}]}

    # Cleanup

    def cleanup(self) -> None:
        if hasattr(self, "mesh") and self.mesh:
            self.mesh.stop()
        if self._world:
            for r in self._world.robots.values():
                r.policy_running = False
            self._world = None
        self._close_viewer()
        # T4: close main-thread renderers before dropping the TLS object.
        # Renderers created on worker threads release their GL contexts
        # when those threads terminate; calling close() cross-thread
        # SIGSEGVs in cgl.free(), so we stay on main.
        self._close_main_thread_renderers()
        if hasattr(self, "_renderer_tls"):
            self._renderer_tls = threading.local()
        self._executor.shutdown(wait=False)
        self._shutdown_event.set()

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
