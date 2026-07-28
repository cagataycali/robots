"""Isaac Sim simulation backend -- GPU-native SimEngine implementation.

Contains :class:`IsaacSimulation`, the ``SimEngine`` implementation for
NVIDIA Isaac Sim. All heavy omni/Isaac imports are lazy; rendering uses
Isaac's RTX pipeline; SimulationApp is a process-wide singleton (never
create more than one).

Thread safety:
    - ``step()``, ``send_action()``, and ``get_observation()`` acquire
      ``self._lock`` to prevent data races
    - ``step()`` must not run concurrently with ``add_robot()``

Environment variables:
    - STRANDS_ISAAC_HEADLESS: Override headless mode (true/false)
    - STRANDS_ISAAC_RTX_PATHTRACING: Enable RTX pathtracing (true/false)
    - STRANDS_ISAAC_NUCLEUS_URL: Override Nucleus asset server URL
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.recording import IsaacRecordingMixin
from strands_robots.utils import positive_whole_number_error

if TYPE_CHECKING:
    from strands_robots.rendering import CameraParams

logger = logging.getLogger(__name__)

# Minimum NATIVE render width for RTX cameras. Isaac's DLSS upscaler
# renders internally at ~half the output width; below ~300 px internal
# resolution it falls back to a temporal-accumulation path that smears a
# moving arm into a translucent "ghost". Rendering at >= 640 px wide stays
# above that threshold; captured frames are downscaled to the caller's
# requested size before return.
_MIN_RENDER_PX = 640


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion (USD convention) to a ``(3, 3)`` rotation matrix."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _env_int(name: str, default: int) -> int:
    """Read a small positive int from the environment (fallback to ``default``)."""
    try:
        v = int(os.environ.get(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment (fallback to ``default``)."""
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


class SimulationAppLaunchConfig(TypedDict, total=False):
    """Typed shape for ``omni.isaac.kit.SimulationApp`` launch config.

    All keys optional; SimulationApp accepts an open-ended dict and any
    additional keys are forwarded to Kit unchanged. Notable semantics:
    ``headless`` must be True on cloud / CI runners; ``renderer`` is
    ``"RayTracedLighting"`` or ``"PathTracing"``; ``physics_gpu`` /
    ``active_gpu`` are CUDA device indices for PhysX / rendering;
    ``width`` / ``height`` are the viewport resolution in pixels;
    ``sync_loads`` blocks until USD assets finish loading;
    ``anti_aliasing`` is a 0-3 level.
    """

    headless: bool
    renderer: str
    width: int
    height: int
    physics_gpu: int
    active_gpu: int
    multi_gpu: bool
    sync_loads: bool
    hide_ui: bool
    anti_aliasing: int


# Shape-name aliases accepted by add_object. ``"cuboid"`` mirrors Isaac's
# ``DynamicCuboid`` / ``FixedCuboid`` class names and normalizes to the
# canonical ``"box"``.
_SHAPE_ALIASES: dict[str, str] = {"cuboid": "box"}


def _rgb_png_block(rgb: np.ndarray) -> dict[str, Any] | None:
    """Encode an RGB ndarray as a render ``content[].image`` PNG block.

    Raw PNG bytes go in ``source.bytes`` (NOT base64 -- the Bedrock
    Converse API base64-encodes on the wire and rejects a pre-encoded
    string). Mirrors the MuJoCo backend so the shared
    ``PolicyRunner._extract_frame_ndarray`` can pull frames for video
    recording. Returns ``None`` if PIL is unavailable or encoding fails,
    so ``render()`` degrades to the legacy rgb-only envelope rather than
    raising (PIL stays a lazy import).
    """
    try:
        import io

        from PIL import Image  # lazy: keep heavy import out of module load

        arr = np.asarray(rgb)[..., :3].astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}}
    except (ImportError, ValueError, OSError, TypeError, AttributeError) as e:
        # PIL absent (ImportError) or encode failure -- never let frame
        # telemetry break render; mirrors the render() except-clause shape.
        logger.warning("render: PNG frame encode failed (%s); content[].image omitted", e)
        return None


# Module-level singleton tracking for SimulationApp
_SIMULATION_APP: Any = None
_SIMULATION_APP_LOCK = threading.Lock()


def _get_or_create_simulation_app(
    headless: bool = True,
    launch_config: SimulationAppLaunchConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create the process-wide SimulationApp singleton.

    Isaac Sim's SimulationApp can only be created ONCE per process; this
    function enforces that. ``launch_config`` (see
    :class:`SimulationAppLaunchConfig`) is the base dict, ``**kwargs`` is
    an escape hatch merged on top, and the explicit ``headless`` argument
    always wins over any ``"headless"`` key.

    Raises
    ------
    ImportError
        If no SimulationApp entry point is available.
    """
    global _SIMULATION_APP

    with _SIMULATION_APP_LOCK:
        if _SIMULATION_APP is not None:
            return _SIMULATION_APP

        try:
            # Isaac Sim 4.5+: ``isaacsim.SimulationApp`` is the supported
            # entry point; the legacy ``omni.isaac.kit.SimulationApp`` may
            # not exist on a pip-only install. Try modern first, fall back
            # to legacy for older builds and CI mocks.
            try:
                from isaacsim import SimulationApp  # type: ignore[import-not-found]
            except ImportError:
                from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]
        except ImportError as e:
            from strands_robots.simulation.isaac._install import not_available_import_error

            raise ImportError(not_available_import_error()) from e

        # Layer order: typed launch_config base, then **kwargs escape hatch,
        # then explicit headless argument (always wins so the caller's
        # intent is unambiguous).
        merged: dict[str, Any] = dict(launch_config or {})
        merged.update(kwargs)
        merged["headless"] = headless
        _SIMULATION_APP = SimulationApp(merged)
        logger.info(
            "SimulationApp created (headless=%s). Note: this is a process-wide singleton.",
            headless,
        )
        return _SIMULATION_APP


# ----------------------------------------------------------------------------
# Dual-namespace import note
# ----------------------------------------------------------------------------
#
# Isaac Sim ships every runtime extension under TWO namespaces: the legacy
# ``omni.isaac.*`` tree (4.x, deprecated shims on 4.5/5.x) and the modern
# ``isaacsim.*`` tree (supported on Isaac Sim 6.0). Every lazy import in
# this module tries the ``isaacsim.*`` location first and falls back to
# ``omni.isaac.*`` so 4.x installs keep working. Namespace map:
#
#   omni.isaac.core.World              -> isaacsim.core.api.World
#   omni.isaac.core.objects.*          -> isaacsim.core.api.objects.*
#   omni.isaac.sensor.Camera           -> isaacsim.sensors.camera.Camera
#   omni.isaac.core.articulations.*    -> isaacsim.core.prims.SingleArticulation
#                                         (see ``_import_articulation_cls``)
#   omni.isaac.core.utils.{prims,
#       stage,viewports}               -> isaacsim.core.utils.{prims,stage,viewports}
#   omni.importer.urdf                 -> isaacsim.asset.importer.urdf
#
# ``import omni.usd`` is NOT renamed (it stays under ``omni.*`` on 6.0).


def _accepts_config_kw(cls: Any) -> bool:
    """True if ``cls.__init__`` accepts a ``config`` keyword argument."""
    try:
        import inspect

        return "config" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return True  # assume yes; the call site falls back to no-arg on TypeError


def _coerce_prim_path(res: Any) -> str:
    """Normalise a URDF-import return value to a USD prim-path string.

    Isaac Sim 6.0's ``URDFImporter.import_urdf()`` may return the prim path
    directly, a ``(status, path)`` tuple, or an object exposing ``.prim_path`` /
    ``.path``. Handle the common shapes; return ``""`` if none match.
    """
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, (tuple, list)):
        for item in res:
            p = _coerce_prim_path(item)
            if p:
                return p
        return ""
    for attr in ("prim_path", "path", "stage_path", "default_prim_path"):
        val = getattr(res, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


def _import_articulation_cls() -> Any:
    """Resolve the single-prim articulation wrapper across Isaac versions.

    Probes the modern 6.0 locations (``isaacsim.core.api.articulations``,
    then ``isaacsim.core.prims``) before the legacy 4.x
    ``omni.isaac.core.articulations`` path. Returns the class object;
    raises ``ImportError`` only if no known location resolves.
    """
    # 1. Isaac Sim 6.0 high-level API (keeps the ``Articulation`` name).
    try:
        from isaacsim.core.api.articulations import (  # type: ignore[import-not-found]
            Articulation,
        )

        return Articulation
    except ImportError:
        pass
    # 2. Isaac Sim 6.0 single-prim view: isaacsim.core.prims.SingleArticulation
    try:
        from isaacsim.core.prims import (  # type: ignore[import-not-found]
            SingleArticulation,
        )

        return SingleArticulation
    except ImportError:
        pass
    # 3. Some 6.0 builds keep an ``Articulation`` alias under core.prims.
    try:
        from isaacsim.core.prims import (  # type: ignore[import-not-found]
            Articulation,
        )

        return Articulation
    except ImportError:
        pass
    # 4. Legacy 4.x fallback.
    from omni.isaac.core.articulations import (  # type: ignore[import-not-found]
        Articulation,
    )

    return Articulation


class _RobotState:
    """Internal bookkeeping for a robot in the Isaac simulation."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        joint_names: list[str],
        articulation: Any = None,
        actual_prim_path: str | None = None,
        data_config: str | None = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.joint_names = joint_names
        self.articulation = articulation
        # Registry data-config the robot was added under (e.g. ``so100``);
        # recorded as the LeRobotDataset ``robot_type``.
        self.data_config = data_config
        # The prim path the URDF importer / USD reference actually placed
        # the robot at, which can differ from ``prim_path`` when the
        # importer ignores the requested destination. Used by
        # ``gripper_frame_pose`` to walk the actual robot subtree.
        self.actual_prim_path = actual_prim_path or prim_path


def _cameras_recording_option_error(
    method: str,
    fps: Any,
    max_frames_per_camera: Any,
) -> dict[str, Any] | None:
    """Reject a rollout-video option the Isaac recorder cannot honor.

    Pre-flight guard for :meth:`IsaacSimulation.start_cameras_recording`,
    mirroring the MuJoCo backend's guard of the same name
    (:func:`strands_robots.simulation.mujoco.rendering._cameras_recording_option_error`)
    against the one shared domain
    (:func:`~strands_robots.utils.positive_whole_number_error`), so the two
    recording surfaces cannot disagree on what a usable ``fps`` is. Isaac takes
    no ``width``/``height`` here - each camera carries its own resolution from
    :meth:`IsaacSimulation.add_camera` - so only the two frame counts are
    checked.

    Refusing at ``start`` is what keeps the flush honest: ``fps`` is stored in
    the recording state and handed to
    :func:`~strands_robots.rendering.encode_clip` by
    :meth:`IsaacSimulation.stop_cameras_recording`, which refuses a rate it
    cannot encode at. Validating only at flush time would surface the mistake
    after a whole rollout's frames had been buffered, and
    ``max_frames_per_camera=0`` would drop every frame while both calls still
    reported success.

    Args:
        method: Public method name, used to prefix the error message.
        fps: Encoded MP4 frame rate.
        max_frames_per_camera: In-memory per-camera frame cap.

    Returns:
        A structured ``{"status": "error", ...}`` dict naming the first
        offending parameter, or ``None`` when both options are usable.
    """
    for param, value in (("fps", fps), ("max_frames_per_camera", max_frames_per_camera)):
        if text := positive_whole_number_error(value, param, method):
            return {"status": "error", "content": [{"text": text}]}
    return None


class _CameraState:
    """Internal bookkeeping for a camera in the Isaac simulation."""

    def __init__(self, name: str, prim_path: str, width: int, height: int):
        self.name = name
        self.prim_path = prim_path
        self.width = width
        self.height = height
        self.handle: Any = None


class _ObjectState:
    """Internal bookkeeping for an object (shape primitive) in the Isaac simulation.

    ``handle`` is the Isaac ``{Dynamic,Fixed}*`` wrapper registered with
    ``world.scene.add()``; held so :meth:`IsaacSimulation.remove_object`
    doesn't have to round-trip through ``world.scene.get_object()`` (which
    can raise on a torn-down stage).
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        shape: str,
        is_static: bool,
        handle: Any = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.shape = shape
        self.is_static = is_static
        self.handle = handle


class IsaacSimulation(IsaacRecordingMixin, SimEngine):
    """GPU-native simulation backend built on NVIDIA Isaac Sim.

    Implements the ``SimEngine`` ABC. Provides photorealistic rendering,
    RTX sensors, USD scene management, and fleet replication via Cloner.
    LeRobotDataset recording (``start_recording`` / ``save_episode`` /
    ``stop_recording`` / ``stream_dataset``) comes from
    :class:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin`,
    matching the MuJoCo and Newton backends.

    Parameters
    ----------
    config : IsaacConfig or None
        Configuration. If None, defaults are used.
    **kwargs
        Shortcut kwargs merged into config (e.g. ``num_envs=1024``).

    Examples
    --------
    >>> sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    >>> ok, msg = IsaacSimulation.is_available()
    >>> if ok:
    ...     sim.create_world()
    ...     sim.add_robot("so100")
    ...     sim.step(100)
    ...     sim.destroy()
    """

    def __init__(self, config: IsaacConfig | None = None, **kwargs: Any) -> None:
        # Merge shortcut kwargs into config. Unknown kwargs are rejected
        # eagerly so a typo surfaces at construction time. A small
        # allow-list of legacy example-adapter kwargs (tool_name,
        # default_timestep/width/height) is accepted for backward compat
        # and kept off the IsaacConfig dataclass.
        import dataclasses

        # Pull the legacy shortcuts out of ``kwargs`` before strict
        # IsaacConfig kwarg-validation runs.
        legacy_tool_name = kwargs.pop("tool_name", "isaac")
        legacy_default_timestep = kwargs.pop("default_timestep", None)
        legacy_default_width = kwargs.pop("default_width", None)
        legacy_default_height = kwargs.pop("default_height", None)

        if config is None:
            # IsaacConfig is a dataclass; an unknown kwarg raises TypeError
            # naturally, matching the strict branch below.
            config = IsaacConfig(**kwargs)
        elif kwargs:
            fields = {f.name for f in dataclasses.fields(config)}
            unknown = sorted(set(kwargs) - fields)
            if unknown:
                raise TypeError(
                    f"IsaacSimulation got unexpected kwargs: {unknown}. Known IsaacConfig fields: {sorted(fields)}."
                )
            config = dataclasses.replace(config, **kwargs)
        # Legacy shortcuts map onto the canonical physics_dt /
        # camera_width / camera_height fields (single source of truth).
        if legacy_default_timestep is not None:
            config = dataclasses.replace(config, physics_dt=float(legacy_default_timestep))
        if legacy_default_width is not None:
            config = dataclasses.replace(config, camera_width=int(legacy_default_width))
        if legacy_default_height is not None:
            config = dataclasses.replace(config, camera_height=int(legacy_default_height))
        self._config = config
        # Tool-name is informational; some Strands tooling renders it.
        self.tool_name = legacy_tool_name

        # Simulation state (all lazy-initialized)
        self._app: Any = None
        self._world: Any = None

        # World state
        self._world_created = False
        self._replicated = False
        self._num_envs_active = 1
        self._sim_time = 0.0
        self._step_count = 0

        # Entity tracking
        self._robots: dict[str, _RobotState] = {}
        self._cameras: dict[str, _CameraState] = {}
        self._objects: dict[str, _ObjectState] = {}
        self._prim_registry: list[str] = []  # track all created prims for cleanup
        # Objects realized by load_scene, kept separate from _objects so a
        # per-episode reload clears only the prior scene's prims.
        self._scene_objects: set[str] = set()
        # Per-camera output size (RTX cameras render at >= _MIN_RENDER_PX
        # wide so DLSS doesn't ghost a moving arm; captured frames are
        # downscaled to the size the caller asked for before return).
        self._cam_out_size: dict[str, tuple[int, int]] = {}
        # Synchronous rollout-video recorder state (set by
        # start_cameras_recording, cleared by stop_cameras_recording).
        self._cams_rec_state: dict[str, Any] | None = None

        # LeRobotDataset recording state seam: MuJoCo/Newton keep this dict
        # on their SimWorld; Isaac's engine owns it directly and
        # IsaacRecordingMixin._recording_state() returns it. Reset by
        # destroy().
        self._recording_state_dict: dict[str, Any] = {}

        # Thread safety
        self._lock = threading.RLock()

        # --- Main-thread pump (for off-main-thread callers, e.g. Gradio).
        # Isaac's renderer + physics may only be driven from the thread
        # that created SimulationApp. When ``run_pump_forever`` is engaged
        # the main thread runs ``pump()`` (steps + renders + caches frames
        # and joint state); worker-thread reads return the cache and
        # worker-thread actions are enqueued. Calls made ON the owning
        # thread run inline (no queue).
        self._main_tid = threading.get_ident()
        self._action_q: queue.Queue = queue.Queue()
        self._main_jobs: queue.Queue = queue.Queue()
        self._frame_cache: dict[str, Any] = {}
        self._joint_cache: dict[str, dict[str, float]] = {}
        self._pump_running = False  # True while run_pump_forever owns the renderer
        self._pump_cameras = True
        # DLSS-convergence tick counts: holding the kinematic arm still
        # for a few RTX render ticks lets the temporal upscaler settle.
        # Env-tunable for headroom on slower GPUs.
        self._record_converge = _env_int("SO101_RECORD_CONVERGE", 6)
        self._idle_converge = _env_int("SO101_IDLE_CONVERGE", 4)
        # Min seconds between IDLE live-preview refreshes; full-speed idle
        # rendering pegs the RTX renderer and starves other threads.
        self._idle_render_period = _env_float("SO101_IDLE_RENDER_PERIOD", 1.0)
        # Render-bearing world steps add_camera takes to warm up a fresh
        # RTX camera's render product (Isaac accumulates no frame until
        # the world is stepped with rendering enabled, so early
        # ``get_rgba()`` calls return a malformed / empty buffer).
        self._camera_warmup_steps = _env_int("STRANDS_ISAAC_CAMERA_WARMUP_STEPS", 10)

        logger.info(
            "IsaacSimulation initialized: num_envs=%d, device=%s, headless=%s",
            config.num_envs,
            config.device,
            config.headless,
        )

    def _on_main_thread(self) -> bool:
        return threading.get_ident() == self._main_tid

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Check if Isaac Sim is available on this system.

        Returns
        -------
        tuple[bool, str | None]
            (available, reason_if_not). If available is True, reason is None.
        """
        # Probe what create_world() actually needs: a SimulationApp entry
        # point in EITHER namespace (legacy ``omni.isaac.kit`` or modern
        # ``isaacsim`` -- some pip installs ship only the modern one). The
        # bare ``omni`` namespace is deliberately NOT probed (a PEP 420
        # namespace package shared by unrelated Omniverse installs, so its
        # presence is not a reliable signal), and neither are submodules
        # deeper than ``isaacsim``, which only resolve AFTER SimulationApp
        # boots the Kit kernel. ``find_spec`` has no import side effects.
        import importlib.util

        try:
            kit_spec = importlib.util.find_spec("omni.isaac.kit")
        except ModuleNotFoundError:
            kit_spec = None
        try:
            isaacsim_spec = importlib.util.find_spec("isaacsim")
        except ModuleNotFoundError:
            isaacsim_spec = None
        if kit_spec is None and isaacsim_spec is None:
            from strands_robots.simulation.isaac._install import not_importable_reason

            return False, not_importable_reason()

        # Isaac requires CUDA
        try:
            import torch

            if not torch.cuda.is_available():
                return False, ("CUDA device not detected. Isaac Sim requires an NVIDIA GPU with CUDA support.")
        except ImportError:
            return False, ("PyTorch not installed. Isaac Sim requires torch with CUDA support.")

        return True, None

    @property
    def config(self) -> IsaacConfig:
        """Current configuration (read-only)."""
        return self._config

    # --- SimEngine: World Lifecycle ----------------------------------------

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        """Create a new simulation world in Isaac Sim.

        Initializes the SimulationApp (singleton), creates a USD stage,
        configures physics, and optionally adds a ground plane.

        Parameters
        ----------
        timestep : float, optional
            Override physics_dt from config (seconds).
        gravity : list[float], optional
            Override gravity vector [gx, gy, gz]. Only Z-aligned gravity
            is supported; anything else is rejected with a structured
            error.
        ground_plane : bool
            Whether to add a ground plane. Default True.
        terrain : str, optional
            Heightfield terrain kind. Not supported on the Isaac backend;
            a non-None value is rejected with an actionable error rather
            than raised or silently ignored (per the
            :class:`~strands_robots.simulation.base.SimEngine`
            ``create_world`` contract).
        difficulty : float
            Terrain elevation scale; inert without terrain, so a
            non-default (``!= 1.0``) value is rejected with an actionable
            error rather than silently having no effect.

        Returns
        -------
        dict
            Status dict with world info.
        """
        if terrain is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"terrain={terrain!r} is not supported on the Isaac backend yet "
                            "(heightfield terrain, e.g. 'rough'/'stairs'/'pyramid'/'slope', is "
                            "currently MuJoCo-only); use create_simulation(backend='mujoco') for "
                            "terrain, or omit terrain for a flat ground plane."
                        )
                    }
                ],
            }
        # Reject a non-default difficulty rather than silently ignoring it:
        # Isaac has no heightfield terrain for it to scale.
        if float(difficulty) != 1.0:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"difficulty={difficulty!r} has no effect on the Isaac backend "
                            "(it scales a heightfield terrain's elevation, and this backend "
                            "has no heightfield terrain); use create_simulation(backend='mujoco') "
                            "for a terrain curriculum, or omit difficulty for a flat ground plane."
                        )
                    }
                ],
            }
        # A world must not be built around a dt the integrator cannot honor
        # (negative, zero, nan): physics_dt drives every stage step, so an
        # unusable value corrupts the world rather than one call.
        effective_timestep = self._config.physics_dt if timestep is None else timestep
        timestep_param = "physics_dt" if timestep is None else "timestep"
        if err := self._validate_timestep(effective_timestep, "create_world", timestep_param):
            return err
        # Isaac's ``PhysicsContext.set_gravity`` takes a single signed
        # scalar (gravity along -Z); validate up front and reject any
        # vector the backend cannot honour rather than silently reducing
        # it to its z-component.
        if gravity is not None:
            if isinstance(gravity, (list, tuple)):
                if len(gravity) != 3:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"create_world: gravity vector must have 3 components [gx, gy, gz], "
                                    f"got {len(gravity)}."
                                )
                            }
                        ],
                    }
                try:
                    gvec = [float(g) for g in gravity]
                except (TypeError, ValueError):
                    return {
                        "status": "error",
                        "content": [{"text": f"create_world: gravity components must be numbers, got {gravity!r}."}],
                    }
                if not all(np.isfinite(gvec)):
                    return {
                        "status": "error",
                        "content": [{"text": f"create_world: gravity components must be finite, got {gravity!r}."}],
                    }
                if gvec[0] != 0.0 or gvec[1] != 0.0:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"create_world: the Isaac backend only supports Z-aligned gravity "
                                    f"(its PhysicsContext.set_gravity takes a signed scalar); a non-Z-aligned "
                                    f"vector like {gravity!r} cannot be honoured. Pass a scalar or a "
                                    f"[0, 0, gz] vector, or use create_simulation(backend='mujoco') for "
                                    f"arbitrary-direction gravity."
                                )
                            }
                        ],
                    }
            elif isinstance(gravity, (int, float)):
                if not np.isfinite(gravity):
                    return {
                        "status": "error",
                        "content": [{"text": f"create_world: gravity must be finite, got {gravity!r}."}],
                    }
            else:
                return {
                    "status": "error",
                    "content": [
                        {"text": f"create_world: gravity must be a scalar or [gx, gy, gz] vector, got {gravity!r}."}
                    ],
                }
        with self._lock:
            if self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "World already created. Call destroy() first."}],
                }

            try:
                # Create/get SimulationApp singleton
                self._app = _get_or_create_simulation_app(headless=self._config.headless)

                # Now safe to import Isaac core modules (modern path first,
                # legacy 4.x fallback -- see the dual-namespace note).
                try:
                    from isaacsim.core.api import World  # type: ignore[import-not-found]
                except ImportError:
                    from omni.isaac.core import World  # type: ignore[import-not-found]

                dt = timestep if timestep is not None else self._config.physics_dt
                grav = gravity if gravity is not None else list(self._config.gravity)

                # Create World
                self._world = World(
                    stage_units_in_meters=1.0,
                    physics_dt=dt,
                    rendering_dt=self._config.rendering_dt,
                )

                # Set gravity
                # Isaac Sim 5.1: set_gravity takes a scalar magnitude, not a vector.
                # Extract the Z-component (convention: gravity points along -Z).
                gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav
                self._world.get_physics_context().set_gravity(gravity_magnitude)

                # Add ground plane
                if ground_plane and self._config.ground_plane:
                    self._world.scene.add_default_ground_plane()
                    self._prim_registry.append(f"{self._config.stage_path}/defaultGroundPlane")

                # Reset world to initialize
                self._world.reset()

                self._world_created = True
                self._sim_time = 0.0
                self._step_count = 0

                logger.info(
                    "World created: dt=%.5f, gravity=%s, headless=%s",
                    dt,
                    grav,
                    self._config.headless,
                )

                # Structured snapshot alongside the human-readable text so
                # agents can introspect without re-querying get_state().
                world_info = {
                    "physics_dt": dt,
                    "rendering_dt": self._config.rendering_dt,
                    "gravity": list(grav) if isinstance(grav, (list, tuple)) else [0.0, 0.0, float(grav)],
                    "ground_plane": bool(ground_plane and self._config.ground_plane),
                    "stage_path": self._config.stage_path,
                    "stage_units_in_meters": 1.0,
                    "device": self._config.device,
                    "headless": self._config.headless,
                    "render_mode": self._config.render_mode,
                    "num_envs": self._config.num_envs,
                    "num_envs_active": self._num_envs_active,
                    "replicated": self._replicated,
                    "sim_time": self._sim_time,
                    "step_count": self._step_count,
                }

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Isaac Sim world created. "
                                f"dt={dt:.5f}, gravity={grav}, "
                                f"device={self._config.device}, "
                                f"headless={self._config.headless}"
                            ),
                            "json": world_info,
                        }
                    ],
                }

            except ImportError as e:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"Isaac Sim import failed: {e}. Ensure Isaac Sim is installed and accessible.")}
                    ],
                }
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                # Cleanup on partial failure. Narrow to what World() /
                # set_gravity / add_default_ground_plane / reset actually
                # raise: RuntimeError (Carb / sim init), ValueError (USD
                # prim shape mismatches), OSError (USD/Nucleus IO),
                # AttributeError / TypeError (omni surface / signature
                # drift across SDK versions). Programming bugs propagate.
                self._world = None
                logger.error("Failed to create Isaac world: %s", e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create world: {e}"}],
                }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources.

        Note: SimulationApp is NOT shut down (it is process-wide).
        Only the World/Stage are cleared.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world to destroy."}],
                }

            # Capture pre-teardown counts for the structured json payload
            # (get_state() is unavailable after destroy() returns).
            num_robots_released = len(self._robots)
            num_cameras_released = len(self._cameras)
            num_objects_released = len(self._objects)
            num_prims_released = len(self._prim_registry)
            num_envs_released = self._num_envs_active
            sim_time_at_destroy = self._sim_time
            step_count_at_destroy = self._step_count

            try:
                if self._world is not None:
                    self._world.stop()
                    self._world.clear_instance()
                    self._world = None
            except (RuntimeError, OSError, AttributeError) as e:
                # World.stop() / clear_instance() can raise on partial init
                # or on a torn-down stage; AttributeError covers omni surface
                # drift across versions. Logged at WARNING because we still
                # mark the world destroyed below; programming bugs propagate.
                logger.warning("World cleanup warning: %s", e)

            # Clear the USD stage. The SimulationApp singleton outlives
            # destroy(), so World.clear_instance() alone leaves this
            # session's prims on the stage; a subsequent create_world()
            # would then build onto a dirty stage and Isaac auto-suffixes
            # colliding paths, breaking prim-path determinism across
            # destroy()/create_world() cycles. A fresh stage pins prim
            # paths stable run-to-run.
            try:
                import omni.usd  # type: ignore[import-not-found]

                omni.usd.get_context().new_stage()
            except (RuntimeError, OSError, AttributeError, ImportError) as e:
                # new_stage() can raise on a torn-down context or omni surface
                # drift across versions; ImportError covers the no-Isaac path.
                # Logged at WARNING because the world is still marked destroyed
                # below; a stale stage only affects a subsequent create_world().
                logger.warning("Stage clear warning: %s", e)

            # Clear entity tracking
            self._robots.clear()
            self._cameras.clear()
            self._objects.clear()
            self._prim_registry.clear()
            # Drop any in-flight recorder state (buffers reference RTX
            # frames that are meaningless after the stage tears down).
            self._cams_rec_state = None
            # Same for the LeRobotDataset recording session (mirrors
            # MuJoCo/Newton, whose state dict dies with the SimWorld);
            # warn when a session was still open so the data loss is
            # visible.
            if self._recording_state_dict.get("recording", False):
                logger.warning(
                    "destroy() called while a dataset recording was active; the unsaved "
                    "episode buffer is discarded. Call stop_recording() before destroy() "
                    "to flush and finalize the dataset."
                )
            self._recording_state_dict = {}

            # Reset state
            self._world_created = False
            self._replicated = False
            self._num_envs_active = 1
            self._sim_time = 0.0
            self._step_count = 0

            logger.info("World destroyed. SimulationApp remains (process-wide singleton).")

            # Structured snapshot of what teardown released (same json
            # content-block convention as get_state() / create_world()).
            destroy_info = {
                "num_robots_released": num_robots_released,
                "num_cameras_released": num_cameras_released,
                "num_objects_released": num_objects_released,
                "num_prims_released": num_prims_released,
                "num_envs_released": num_envs_released,
                "sim_time_at_destroy": sim_time_at_destroy,
                "step_count_at_destroy": step_count_at_destroy,
                "stage_path": self._config.stage_path,
                "simulation_app_alive": True,  # singleton survives destroy()
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Isaac Sim world destroyed. All resources released. SimulationApp singleton remains active."
                        ),
                        "json": destroy_info,
                    }
                ],
            }

    def reset(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset simulation to initial state.

        Parameters
        ----------
        env_ids : list[int], optional
            Specific environment indices to reset. If None, reset all.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is not None:
                self._world.reset()

            self._sim_time = 0.0
            self._step_count = 0

            if env_ids is None:
                msg = "Full reset complete."
            else:
                msg = f"Partial reset complete for {len(env_ids)} envs."

            return {"status": "success", "content": [{"text": msg}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take. Default 1.

        Returns
        -------
        dict
            Status dict with timing info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is None:
                return {"status": "error", "content": [{"text": "World not initialized."}]}

            t0 = time.perf_counter()

            for _ in range(n_steps):
                self._world.step(render=self._config.render_mode != "headless")
                self._sim_time += self._config.physics_dt
                self._step_count += 1

            elapsed = time.perf_counter() - t0
            steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Stepped {n_steps}x. "
                            f"sim_time={self._sim_time:.4f}s, "
                            f"wall={elapsed * 1000:.1f}ms, "
                            f"{steps_per_sec:.0f} steps/sec"
                        )
                    }
                ],
            }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary.

        Returns
        -------
        dict
            Status dict with state information.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            state_data = {
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "num_envs": self._num_envs_active,
                "num_robots": len(self._robots),
                "num_cameras": len(self._cameras),
                "num_objects": len(self._objects),
                "stage_path": self._config.stage_path,
                "device": self._config.device,
                "headless": self._config.headless,
                "render_mode": self._config.render_mode,
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"State: t={self._sim_time:.4f}s, "
                            f"step={self._step_count}, "
                            f"envs={self._num_envs_active}, "
                            f"robots={len(self._robots)}, "
                            f"cameras={len(self._cameras)}, "
                            f"objects={len(self._objects)}"
                        ),
                        "json": state_data,
                    }
                ],
            }

    # --- SimEngine: Robot Management ----------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        mjcf_path: str | None = None,
        usd_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup).
        urdf_path : str, optional
            Path to URDF file.
        mjcf_path : str, optional
            Not supported (Isaac has no MJCF robot importer); a non-None
            value is rejected with an actionable error rather than being
            silently ignored. Convert the MJCF to URDF/USD, or use the
            MuJoCo backend.
        usd_path : str, optional
            Path to USD file (native Isaac format).
        data_config : str, optional
            Named data config for procedural lookup.
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base quaternion [w, x, y, z]. The Isaac spawn path applies no
            base rotation, so a non-identity quaternion is rejected with
            an actionable error rather than silently dropped; identity
            ``[1, 0, 0, 0]`` (or ``None``) is fine.
        keyframe : str | int, optional
            MuJoCo ``<keyframe>`` pose; not parsed on Isaac, so a
            non-None value is rejected with an actionable error rather
            than raised or silently ignored (per the
            :class:`~strands_robots.simulation.base.SimEngine`
            ``add_robot`` contract).

        Returns
        -------
        dict
            Status dict with robot info.
        """
        if keyframe is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: keyframe={keyframe!r} is not supported on "
                            "the Isaac backend (spawning at a MuJoCo <keyframe> pose "
                            "is currently MuJoCo-only); use "
                            "create_simulation(backend='mujoco') to spawn at a "
                            "keyframe, or omit keyframe for the default zero-pose "
                            "spawn."
                        )
                    }
                ],
            }
        if mjcf_path is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: mjcf_path={mjcf_path!r} is not supported on the Isaac "
                            "backend (it has no MJCF robot importer; it loads USD natively and "
                            "converts URDF). Convert the MJCF to URDF/USD and pass urdf_path/"
                            "usd_path, or use create_simulation(backend='mujoco') to load MJCF."
                        )
                    }
                ],
            }
        # None means identity; only a non-identity quaternion is rejected.
        if orientation is not None and list(orientation) != [1.0, 0.0, 0.0, 0.0]:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: orientation={orientation!r} is not applied on the Isaac "
                            "backend spawn path (the USD/URDF loaders position the articulation but "
                            "ignore base rotation). Omit orientation (or pass identity "
                            "[1, 0, 0, 0]) for the default upright spawn, or use "
                            "create_simulation(backend='mujoco') to spawn at an orientation."
                        )
                    }
                ],
            }
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world created. Call create_world() first."}],
                }

            if name in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' already exists."}],
                }

            if self._replicated:
                return {
                    "status": "error",
                    "content": [{"text": "Cannot add robots after replicate(). Call destroy() first."}],
                }

            pos = position or [0.0, 0.0, 0.0]
            prim_path = f"{self._config.stage_path}/Robots/{name}"

            # Procedural lookup is a *fallback*: an explicit usd_path /
            # urdf_path always wins, so a name colliding with the
            # procedural registry cannot shadow a concrete asset.
            lookup_name = data_config or name
            try:
                from strands_robots.simulation.isaac.procedural import get_procedural_robot

                procedural = get_procedural_robot(lookup_name)
            except ImportError:
                procedural = None

            if procedural is not None and usd_path is None and urdf_path is None:
                # Build procedurally via USD API
                joint_names = procedural.joint_names
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                    data_config=data_config,
                )
                self._robots[name] = robot_state

                logger.info("Added robot '%s' (procedural, %d joints)", name, len(joint_names))
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' added (procedural: {procedural.name}, "
                                f"{len(joint_names)} joints: {joint_names})"
                            )
                        }
                    ],
                }

            elif usd_path is not None:
                # Load from USD (native Isaac format): references the USD
                # into the stage, constructs + initialises an Articulation.
                try:
                    joint_names, articulation = self._load_usd_robot(prim_path, usd_path, pos)
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                    # Cleanup-clause shape mirrors create_world; programming
                    # bugs propagate.
                    logger.error(
                        "Failed to load USD robot '%s' (usd_path=%s): %s",
                        name,
                        usd_path,
                        e,
                    )
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to load USD robot '{name}': {e}"}],
                    }

                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                    articulation=articulation,
                    actual_prim_path=getattr(articulation, "_strands_actual_prim_path", None),
                    data_config=data_config,
                )
                self._robots[name] = robot_state

                logger.info(
                    "Added robot '%s' (USD: %s, %d joints, articulation=%s)",
                    name,
                    usd_path,
                    len(joint_names),
                    "wired" if articulation is not None else "phase1",
                )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (f"Robot '{name}' added (USD: {usd_path}, {len(joint_names)} joints)"),
                            "json": {
                                "name": name,
                                "prim_path": prim_path,
                                "usd_path": usd_path,
                                "joint_names": joint_names,
                                "joint_count": len(joint_names),
                                "position": pos,
                                "articulation_wired": articulation is not None,
                            },
                        }
                    ],
                }

            elif urdf_path is not None:
                # Convert URDF to USD and load (runs the URDF importer and
                # constructs an Articulation).
                try:
                    joint_names, articulation = self._load_urdf_robot(prim_path, urdf_path, pos)
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                    # Cleanup-clause shape mirrors the USD branch above.
                    logger.error(
                        "Failed to load URDF robot '%s' (urdf_path=%s): %s",
                        name,
                        urdf_path,
                        e,
                    )
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to load URDF robot '{name}': {e}"}],
                    }

                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                    articulation=articulation,
                    actual_prim_path=getattr(articulation, "_strands_actual_prim_path", None),
                    data_config=data_config,
                )
                self._robots[name] = robot_state

                logger.info(
                    "Added robot '%s' (URDF: %s, %d joints, articulation=%s)",
                    name,
                    urdf_path,
                    len(joint_names),
                    "wired" if articulation is not None else "phase1",
                )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (f"Robot '{name}' added (URDF: {urdf_path}, {len(joint_names)} joints)"),
                            "json": {
                                "name": name,
                                "prim_path": prim_path,
                                "urdf_path": urdf_path,
                                "joint_names": joint_names,
                                "joint_count": len(joint_names),
                                "position": pos,
                                "articulation_wired": articulation is not None,
                            },
                        }
                    ],
                }

            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{lookup_name}' not found in procedural registry "
                                "and no usd_path/urdf_path provided. "
                                "Available procedural robots: so100, panda, unitree_g1"
                            )
                        }
                    ],
                }

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
        material: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object (shape primitive) to the scene.

        Instantiates the USD prim via Isaac's ``{Dynamic,Fixed}*`` object
        wrappers and registers it with ``world.scene.add()``.

        Parameters
        ----------
        name : str
            Object identifier. Must be unique; a duplicate is rejected
            with a structured error envelope.
        shape : str
            ``"box"`` (default), ``"sphere"``, ``"capsule"``,
            ``"cylinder"``. ``"cuboid"`` is accepted as an alias and
            normalizes to ``"box"`` (the value reported in the result
            ``json``). Anything else returns a structured error listing
            the valid set.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[0.0, 0.0, 0.5]``.
        orientation : list[float], optional
            Quaternion ``[w, x, y, z]``. Default identity.
        size : list[float], optional
            Shape dimensions in meters; ``scale`` is accepted as an alias
            (explicit ``size`` wins). Conventions per shape:

            * ``box``:      ``[width, height, depth]`` (default ``[0.05, 0.05, 0.05]``).
            * ``sphere``:   ``[radius]`` (default ``[0.05]``).
            * ``cylinder``: ``[radius, height]`` (default ``[0.05, 0.10]``).
            * ``capsule``:  ``[radius, height]`` (default ``[0.05, 0.10]``).

            Lists shorter than the convention fall back to defaults for
            the missing trailing components.
        color : list[float], optional
            RGB ``[r, g, b]`` in ``[0, 1]``; RGBA accepted, alpha dropped.
            ``None`` -> default white.
        mass : float
            Mass in kg. Default 0.1. Ignored when ``is_static=True``.
        is_static : bool
            ``True`` -> ``Fixed*`` prim pinned in space; ``False``
            (default) -> ``Dynamic*`` prim with physics.
        mesh_path : str, optional
            Custom meshes are not supported on Isaac; a non-``None`` value
            is rejected with an actionable error rather than silently
            ignored (per the
            :class:`~strands_robots.simulation.base.SimEngine`
            ``add_object`` contract).
        material : dict, optional
            Not supported on Isaac; a non-``None`` value is rejected
            loudly rather than silently dropped.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope; ``json`` carries the resolved ``prim_path``,
            ``shape``, ``position``, ``orientation``, ``size``, ``mass``,
            and ``is_static``.
        """
        if material is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_object: material= is not supported on the "
                            "Isaac backend yet (matte/textured surfaces); use "
                            "create_simulation(backend='mujoco') for materials, "
                            "or omit material for a flat color."
                        )
                    }
                ],
            }
        if mesh_path is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_object: mesh_path=/custom mesh objects are not "
                            "supported on the Isaac backend yet; use "
                            "create_simulation(backend='mujoco') for meshes, or a "
                            "primitive shape (box/sphere/capsule/cylinder)."
                        )
                    }
                ],
            }
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Normalize shape aliases (``"cuboid"`` -> canonical ``"box"``).
            shape = _SHAPE_ALIASES.get(shape, shape)

            # Validate shape
            valid_shapes = ("box", "sphere", "capsule", "cylinder")
            if shape not in valid_shapes:
                accepted = valid_shapes + tuple(_SHAPE_ALIASES)
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown shape: {shape!r}. Valid: {accepted}"}],
                }

            if name in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' already exists."}],
                }

            # ``scale`` is an alias for ``size``; explicit ``size`` wins.
            scale_alias = kwargs.pop("scale", None)
            if size is None and scale_alias is not None:
                size = scale_alias

            pos = list(position) if position is not None else [0.0, 0.0, 0.5]
            orient = list(orientation) if orientation is not None else [1.0, 0.0, 0.0, 0.0]
            size_in = list(size) if size is not None else None
            prim_path = f"{self._config.stage_path}/Objects/{name}"

            try:
                handle, resolved_size = self._construct_shape_prim(
                    shape=shape,
                    prim_path=prim_path,
                    name=name,
                    position=pos,
                    orientation=orient,
                    size=size_in,
                    color=color,
                    mass=mass,
                    is_static=is_static,
                )
                # ``world.scene.add`` registers the wrapper so
                # ``world.reset()`` re-initialises it; we keep our own ref
                # in ``_objects[name]`` so ``remove_object`` doesn't have
                # to round-trip through ``scene.get_object`` later.
                self._world.scene.add(handle)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Cleanup-clause shape mirrors create_world; programming
                # bugs propagate. The bare ``Exception`` from
                # ``omni.physics.tensors`` on an eager Dynamic* velocity
                # query is NOT caught here -- ``_construct_shape_prim``
                # prevents it up front by stopping the timeline before
                # constructing dynamic prims, keeping this clause free of
                # a bare ``except Exception``.
                logger.error(
                    "Failed to add object '%s' (shape=%s, static=%s): %s",
                    name,
                    shape,
                    is_static,
                    e,
                )
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add object '{name}' ({shape}): {e}"}],
                }

            self._prim_registry.append(prim_path)
            self._objects[name] = _ObjectState(
                name=name,
                prim_path=prim_path,
                shape=shape,
                is_static=is_static,
                handle=handle,
            )

            obj_info = {
                "name": name,
                "prim_path": prim_path,
                "shape": shape,
                "position": pos,
                "orientation": orient,
                "size": resolved_size,
                "mass": float(mass) if not is_static else 0.0,
                "is_static": bool(is_static),
            }
            logger.info(
                "Added object '%s' (shape=%s, pos=%s, mass=%.3f, static=%s)",
                name,
                shape,
                pos,
                mass,
                is_static,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Object '{name}' added (shape={shape}, pos={pos}).",
                        "json": obj_info,
                    }
                ],
            }

    def _construct_shape_prim(
        self,
        *,
        shape: str,
        prim_path: str,
        name: str,
        position: list[float],
        orientation: list[float],
        size: list[float] | None,
        color: list[float] | None,
        mass: float,
        is_static: bool,
    ) -> tuple[Any, list[float]]:
        """Construct a shape prim, deferring physics init for dynamic bodies.

        Isaac's ``Dynamic*`` constructors run an eager velocity query
        during ``__init__`` (``RigidPrim._on_physics_ready`` ->
        ``omni.physics.tensors``) which raises a bare ``Exception`` when
        the new prim is not yet part of the physics-tensor view. We
        *prevent* that rather than catch it (a bare ``except Exception``
        is forbidden in this module): before constructing any ``Dynamic*``
        prim the timeline is stopped unconditionally -- ``timeline.stop()``
        is idempotent, and probe-gating proved unreliable because the
        probe checks a different tensor-view handle than ``RigidPrim``
        keys off. Stopping clears the tensor view so the eager query is
        skipped; the prim initialises cleanly on the next ``world.reset()``.
        Static (``Fixed*``) prims never take that path and leave the
        timeline untouched.

        Returns the same ``(handle, resolved_size)`` tuple as
        :meth:`_create_shape_prim`.
        """
        if not is_static:
            logger.info(
                "Stopping timeline before constructing dynamic prim '%s' so "
                "RigidPrim.__init__ skips its eager velocity query for a prim not "
                "yet in the tensor view (#159). The prim initialises on the next "
                "reset(). Idempotent if the timeline is already stopped.",
                name,
            )
            self._stop_timeline_for_deferred_physics()
        return self._create_shape_prim(
            shape=shape,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            size=size,
            color=color,
            mass=mass,
            is_static=is_static,
        )

    @staticmethod
    def _stop_timeline_for_deferred_physics() -> None:
        """Stop the Isaac timeline so the physics-tensor view is cleared.

        After this returns the physics-tensor view is torn down, so
        ``RigidPrim.__init__`` skips its eager ``_on_physics_ready``
        velocity query and a freshly constructed ``Dynamic*`` prim
        initialises only on the next ``world.reset()``. Best-effort: a
        missing ``omni.timeline`` (partial Isaac install) is logged and
        ignored.
        """
        try:
            import omni.timeline  # type: ignore[import-not-found]

            omni.timeline.get_timeline_interface().stop()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("Could not stop timeline to defer physics init: %s", e)

    def _create_shape_prim(
        self,
        *,
        shape: str,
        prim_path: str,
        name: str,
        position: list[float],
        orientation: list[float],
        size: list[float] | None,
        color: list[float] | None,
        mass: float,
        is_static: bool,
    ) -> tuple[Any, list[float]]:
        """Construct the Isaac object shape wrapper.

        Returns the handle plus the resolved ``size`` list (defaults
        applied per shape) so :meth:`add_object` can surface the
        actually-used dimensions. Lazy-imports the Isaac constructors
        (modern path first, legacy 4.x fallback) so the module loads
        without Isaac Sim installed.
        """
        import numpy as np  # type: ignore[import-not-found]

        try:
            from isaacsim.core.api.objects import (  # type: ignore[import-not-found]
                DynamicCapsule,
                DynamicCuboid,
                DynamicCylinder,
                DynamicSphere,
                FixedCapsule,
                FixedCuboid,
                FixedCylinder,
                FixedSphere,
            )
        except ImportError:
            from omni.isaac.core.objects import (  # type: ignore[import-not-found]
                DynamicCapsule,
                DynamicCuboid,
                DynamicCylinder,
                DynamicSphere,
                FixedCapsule,
                FixedCuboid,
                FixedCylinder,
                FixedSphere,
            )

        common: dict[str, Any] = {
            "prim_path": prim_path,
            "name": name,
            "position": np.asarray(position, dtype=float),
            "orientation": np.asarray(orientation, dtype=float),
        }
        if color is not None:
            # RGBA -> RGB: Isaac's primitive constructors take a 3-vector
            # color; alpha would silently raise a shape mismatch deeper
            # in USD. Truncate here so RGBA-style examples (e.g. the #15
            # sketch's ``[1, 0, 0, 1]``) work transparently.
            rgb = list(color)[:3]
            common["color"] = np.asarray(rgb, dtype=float)
        if not is_static:
            common["mass"] = float(mass)

        if shape == "box":
            cls = FixedCuboid if is_static else DynamicCuboid
            # Per-component fallback: lists shorter than the convention
            # fall back to defaults for the missing trailing components.
            size_list = list(size) if size else []
            sx = float(size_list[0]) if len(size_list) >= 1 else 0.05
            sy = float(size_list[1]) if len(size_list) >= 2 else 0.05
            sz = float(size_list[2]) if len(size_list) >= 3 else 0.05
            scale = [sx, sy, sz]
            common["scale"] = np.asarray(scale, dtype=float)
            return cls(**common), scale
        if shape == "sphere":
            cls = FixedSphere if is_static else DynamicSphere
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            return cls(radius=radius, **common), [radius]
        if shape == "cylinder":
            cls = FixedCylinder if is_static else DynamicCylinder
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        if shape == "capsule":
            cls = FixedCapsule if is_static else DynamicCapsule
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        # Unreachable: shape was validated by add_object before this call;
        # raise loudly if a future caller bypasses that guard.
        raise ValueError(f"Unknown shape: {shape!r}")

    # --- SimEngine: Scene loading -------------------------------------------

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Realize a LIBERO/BDDL task scene as USD prims on the Isaac stage.

        ``scene_path`` is a robosuite-compiled LIBERO MJCF XML.
        ``load_mjcf_scene_objects`` walks the ``<worldbody>``, skipping the
        floor (created by :meth:`create_world`) and the robot (loaded via
        :meth:`add_robot`). LIBERO meshes aren't portable to the Isaac
        stage, so each object is approximated by a box primitive sized to
        the AABB of its collision geoms at its MJCF body pose, realized via
        :meth:`add_object` (static fixtures -> ``Fixed*``; movable ->
        ``Dynamic*``).

        Idempotency: a fresh ``load_scene`` first removes objects left over
        from a prior episode's scene (tracked in ``_scene_objects``) so
        per-episode reloads don't accumulate duplicate prims.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope; on success ``json`` carries the realized object
            count and names. Recoverable failures (no world,
            missing/malformed file) return ``{"status": "error", ...}``
            rather than raising.
        """
        from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects

        with self._lock:
            if not self._world_created:
                msg = f"Cannot load scene: no world created. Call create_world() before load_scene({scene_path!r})."
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            if not scene_path or not os.path.exists(scene_path):
                msg = f"Scene file not found: {scene_path!r}"
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            # Parse the MJCF -> a backend-agnostic list of SceneObjects.
            try:
                scene_objects = load_mjcf_scene_objects(scene_path)
            except (FileNotFoundError, ValueError) as e:
                msg = f"Failed to parse LIBERO scene {scene_path!r}: {e}"
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            # Clear any objects realized by a prior load_scene so per-episode
            # reloads are idempotent (no duplicate prims / no "already exists").
            for prior_name in list(self._scene_objects):
                if prior_name in self._objects:
                    self.remove_object(prior_name)
                self._scene_objects.discard(prior_name)

            realized: list[str] = []
            skipped: list[dict[str, Any]] = []
            for obj in scene_objects:
                # ``add_object`` rejects duplicate names; if a manually-added
                # object shadows a scene object, skip it rather than abort.
                if obj.name in self._objects:
                    skipped.append({"name": obj.name, "reason": "name already in use"})
                    continue
                result = self.add_object(
                    name=obj.name,
                    shape="box",
                    position=list(obj.position),
                    orientation=list(obj.quat),
                    size=list(obj.size),
                    mass=0.1,
                    is_static=obj.is_static,
                )
                if result.get("status") == "success":
                    realized.append(obj.name)
                    self._scene_objects.add(obj.name)
                else:
                    text = (result.get("content") or [{}])[0].get("text", "")
                    skipped.append({"name": obj.name, "reason": text})

            summary = (
                f"Loaded LIBERO scene from {os.path.basename(scene_path)}: "
                f"realized {len(realized)} object(s) as Isaac stage prims"
            )
            if skipped:
                summary += f" ({len(skipped)} skipped)"
            logger.info("IsaacSimulation.load_scene: %s", summary)
            return {
                "status": "success",
                "content": [
                    {
                        "text": summary,
                        "json": {
                            "scene_path": scene_path,
                            "realized": realized,
                            "skipped": skipped,
                            "object_count": len(realized),
                        },
                    }
                ],
            }

    # --- SimEngine: Introspection / Removal ---------------------------------

    def list_robots(self) -> list[str]:
        """Return robot names in insertion order (empty if none added or after :meth:`destroy`)."""
        with self._lock:
            return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return joint names for ``robot_name`` in articulation order.

        Returns an empty list if ``robot_name`` is not present (matches
        the silent-empty convention of :meth:`get_observation`).
        """
        with self._lock:
            if robot_name not in self._robots:
                return []
            return list(self._robots[robot_name].joint_names)

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation.

        Drops the robot's bookkeeping entry and prunes its prims from
        ``self._prim_registry``; the actual USD prim deletion is delegated
        to :meth:`destroy` / world teardown.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}]}`` envelope.
        """
        with self._lock:
            if name not in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' not found."}],
                }
            prim_path = self._robots[name].prim_path
            self._prim_registry = [p for p in self._prim_registry if not p.startswith(prim_path)]
            del self._robots[name]
            logger.info("Removed robot '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Robot '{name}' removed."}],
            }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene.

        Calls ``world.scene.remove_object(name)`` to delete the USD prim,
        then prunes the in-Python registries; the prim is gone from the
        stage when this returns.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}]}`` envelope.
            Returns ``error`` if the object is unknown to ``_objects``
            (the authoritative source).
        """
        with self._lock:
            if name not in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' not found."}],
                }

            prim_path = self._objects[name].prim_path

            # Delete the prim; same cleanup-clause shape as add_object.
            try:
                if self._world is not None:
                    self._world.scene.remove_object(name)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                logger.error("Failed to remove object '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove object '{name}': {e}"}],
                }

            # Drop bookkeeping only after the scene call succeeded so a
            # transient failure leaves a retry-friendly state.
            del self._objects[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed object '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' removed."}],
            }

    # --- SimEngine: Observation / Action ------------------------------------

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get observation for a robot.

        Parameters
        ----------
        robot_name : str, optional
            Robot to observe. Auto-resolves if only one robot exists.
        skip_images : bool
            Skip camera rendering. Default False.

        Returns
        -------
        dict
            Plain observation dict (joint positions keyed by joint name,
            plus camera frames keyed by camera name), NOT the
            ``{"status", "content"}`` envelope. An empty dict indicates
            one of four conditions, each logged before return: world not
            yet created (DEBUG), ambiguous ``robot_name=None`` with
            multiple robots (WARNING), unknown ``robot_name`` (WARNING),
            or an uninitialised Articulation handle (DEBUG).
        """
        with self._lock:
            if not self._world_created:
                # Expected pre-init state; many callers probe before
                # create_world() to feature-detect, so DEBUG-only.
                logger.debug(
                    "get_observation(robot_name=%r): world not yet created",
                    robot_name,
                )
                return {}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    logger.warning(
                        "get_observation(robot_name=None): ambiguous -- "
                        "%d robots present (%s); pass robot_name explicitly. "
                        "Returning empty observation.",
                        len(self._robots),
                        sorted(self._robots),
                    )
                    return {}

            if robot_name not in self._robots:
                logger.warning(
                    "get_observation(robot_name=%r): unknown robot. Known: %s. Returning empty observation.",
                    robot_name,
                    sorted(self._robots),
                )
                return {}

            robot = self._robots[robot_name]
            obs: dict[str, Any] = {}

            # Get joint state from Articulation handle
            if robot.articulation is not None:
                try:
                    joint_positions = robot.articulation.get_joint_positions()
                    if joint_positions is not None:
                        positions = (
                            joint_positions.cpu().numpy()
                            if hasattr(joint_positions, "cpu")
                            else np.array(joint_positions)
                        )
                        for i, jname in enumerate(robot.joint_names):
                            if i < len(positions):
                                obs[jname] = float(positions[i])
                except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                    # Articulation handle may raise RuntimeError on a not-yet
                    # -initialized world, AttributeError on torch-tensor surface
                    # drift, ValueError/TypeError on np coercion. Programming
                    # bugs propagate.
                    logger.debug("Failed to get joint positions: %s", e)

            # Camera frames keyed by camera name (RGB HxWx3 uint8), same
            # shape as the MuJoCo backend. Skipped when ``skip_images`` or
            # in headless render mode. Best-effort per camera: a camera
            # whose RTX product hasn't warmed up is omitted rather than
            # failing the whole observation.
            #
            # Recording override (parity with MuJoCo/Newton): while a
            # dataset recording is active the recorded frames MUST carry
            # the camera images the schema declared, so images are forced
            # on even when the caller passed skip_images=True.
            if skip_images:
                rec_state = self._recording_state()
                if rec_state is not None and rec_state.get("recording", False):
                    skip_images = False
            if not skip_images and self._config.render_mode != "headless":
                # Multi-camera refresh: a single ``world.step(render=True)``
                # reliably refreshes only the PRIMARY render product, so
                # with more than one camera we tick the renderer extra
                # times so every RTX product holds a fresh frame.
                # Single-camera setups skip this to stay fast.
                if len(self._cameras) > 1:
                    self._refresh_all_render_products()
                for cam_name, cam in self._cameras.items():
                    if cam.handle is None:
                        continue
                    try:
                        rgba = cam.handle.get_rgba()
                        # Validate shape BEFORE slicing: a 0-D buffer from
                        # a not-yet-warmed RTX product makes ``[..., :3]``
                        # raise IndexError. Skip such cameras rather than
                        # failing the whole observation.
                        arr = np.asarray(rgba)
                        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
                            obs[cam_name] = arr[..., :3].astype(np.uint8)
                    except (RuntimeError, ValueError, AttributeError, TypeError, IndexError) as e:
                        logger.debug("camera %r frame unavailable: %s", cam_name, e)

            return obs

    def send_action(
        self,
        action: dict[str, Any] | np.ndarray | list,
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action and advance physics.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name.
        robot_name : str, optional
            Robot to control.
        n_substeps : int
            Physics sub-steps after applying action. Default 1.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}]}`` envelope, matching
            the :class:`~strands_robots.simulation.base.SimEngine` contract so
            :class:`~strands_robots.simulation.policy_runner.PolicyRunner` can
            count action failures (it increments ``_action_errors`` when the
            returned ``status`` is ``"error"``). When ``action`` is a dict and
            some keys don't name a joint on ``robot_name``, the ``content`` list
            carries a ``json`` block with ``unresolved_keys`` / ``applied`` so
            callers can self-correct -- mirroring the MuJoCo backend.
        """
        with self._lock:
            if not self._world_created or self._world is None:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                elif not self._robots:
                    return {"status": "error", "content": [{"text": "No robots in the world."}]}
                else:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Multiple robots present; specify robot_name. Available: {sorted(self._robots)}"
                                )
                            }
                        ],
                    }

            if robot_name not in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

            robot = self._robots[robot_name]

            # Convert action to array, tracking dict keys that don't name a
            # joint so unresolved commands surface in the envelope rather than
            # being silently dropped (parity with the MuJoCo backend).
            unresolved: list[str] = []
            action_array: np.ndarray
            # ``joint_indices`` restricts an ``ArticulationAction`` to a subset
            # of the articulation's DOFs; ``None`` addresses every joint. For a
            # dict action we command ONLY the named joints and leave the rest at
            # their current PD targets (parity with the MuJoCo/Newton backends).
            # A full zero-filled ``joint_positions`` vector would instead drive
            # every unnamed joint to 0.0 -- e.g. ``send_action({"gripper": 0.04})``
            # would slam the whole arm to its home pose.
            joint_indices: np.ndarray | None
            if isinstance(action, dict):
                joint_set = set(robot.joint_names)
                unresolved = [k for k in action if k not in joint_set]
                named = [i for i, jname in enumerate(robot.joint_names) if jname in action]
                action_array = np.array(
                    [float(action[robot.joint_names[i]]) for i in named],
                    dtype=np.float32,
                )
                joint_indices = np.array(named, dtype=np.int32)
            elif isinstance(action, np.ndarray):
                action_array = action.astype(np.float32).flatten()
                joint_indices = None
            else:
                action_array = np.array(action, dtype=np.float32)
                joint_indices = None

            # Apply to articulation. Isaac Sim 6.0 drives PD position
            # targets via ``apply_action(ArticulationAction(...))`` (the
            # pre-6.0 ``set_joint_position_targets`` does not exist on the
            # 6.0 class). See ``set_joint_positions`` for the teleport
            # (non-PD) counterpart.
            if robot.articulation is not None and action_array.size > 0:
                try:
                    from isaacsim.core.utils.types import (  # type: ignore[import-not-found]
                        ArticulationAction,
                    )

                    robot.articulation.apply_action(
                        ArticulationAction(joint_positions=action_array, joint_indices=joint_indices)
                    )
                except (RuntimeError, ValueError, AttributeError, ImportError) as e:
                    # Torn-down articulation / shape mismatch / omni surface
                    # drift / isaacsim not importable. Programming bugs
                    # propagate.
                    logger.debug("Failed to set joint targets: %s", e)
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to set joint targets on '{robot_name}': {e}"}],
                    }

            # Step physics. Render on the LAST substep when not headless so the
            # RTX camera render products refresh -> ``get_rgba`` returns a fresh
            # frame for this step (otherwise every recorded frame is identical,
            # i.e. a static video). Intermediate substeps skip render for speed.
            render_on = self._config.render_mode != "headless"
            for i in range(n_substeps):
                last = i == n_substeps - 1
                self._world.step(render=bool(render_on and last))
                self._sim_time += self._config.physics_dt
                self._step_count += 1

        if unresolved:
            applied = [k for k in action if k not in unresolved]
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action partially applied: keys {unresolved} could not be "
                            f"resolved to joints on '{robot_name}'. Applied: {applied}. "
                            f"Valid keys: {robot.joint_names}"
                        )
                    },
                    {"json": {"unresolved_keys": unresolved, "applied": applied}},
                ],
            }

        return {
            "status": "success",
            "content": [{"text": f"Action applied to '{robot_name}', {n_substeps} substeps."}],
        }

    # --- SimEngine: Rendering -----------------------------------------------

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Render a camera view using Isaac Sim's RTX pipeline.

        A camera with a live RTX ``handle`` (and a non-headless render
        mode) yields real frames; otherwise blank frames are returned via
        one of four fallback paths, each tagged in the success envelope's
        text so a caller can tell which path was taken:

        * ``Rendered (headless, no RTX)`` -- headless render mode (most
          CI flows).
        * ``Rendered (no camera)`` -- unknown ``camera_name``.
        * ``Rendered (Phase-1 camera, no RTX handle)`` -- registered
          camera whose ``handle`` is ``None`` (defensive fallback).
        * ``Rendered (RTX <render_mode>)`` -- real frames at the camera's
          resolved resolution (the ``width`` / ``height`` arguments only
          size the blank-frame fallbacks).

        Returns
        -------
        dict
            Standard Strands tool-result envelope carrying ONLY
            ``status`` and ``content`` (the tool-result contract forbids
            extra top-level keys). On success ``content`` holds a
            ``text`` block, a ``{"image": {"format": "png", ...}}`` block
            with raw PNG bytes (matching the MuJoCo backend so the shared
            ``PolicyRunner._extract_frame_ndarray`` can pull frames for
            video recording), and a ``{"json": {...}}`` block with pixel
            stats plus (RTX path) the resolved camera ``resolution``,
            ``prim_path``, and the boolean ``rtx`` flag. The PNG block is
            omitted (and a warning logged) if PIL is unavailable.
            Consumers needing raw ``rgb`` / ``depth`` ndarrays use
            :meth:`get_observation` or the internal :meth:`_render_frame`.
        """
        rgb, depth, meta = self._render_frame(camera_name, width, height)
        if rgb is None:
            # meta carries the structured error text on the failure path.
            return {
                "status": "error",
                "content": [{"text": meta.get("error", "render failed")}],
            }
        content: list[dict[str, Any]] = [{"text": meta.get("text", "")}]
        block = _rgb_png_block(rgb)
        if block is not None:
            content.append(block)
        # Structured telemetry lives INSIDE a content json block, never as
        # extra top-level keys (tool-result contract: only status/content).
        json_block: dict[str, Any] = dict(meta.get("json", {}))
        json_block["pixel_mean"] = float(np.mean(rgb))
        json_block["pixel_variance"] = float(np.var(rgb))
        json_block["camera"] = camera_name
        content.append({"json": json_block})
        return {"status": "success", "content": content}

    def _render_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """Render a camera to raw ``(rgb, depth, meta)`` for internal consumers.

        Numeric-array counterpart to the public :meth:`render` (which
        wraps this into the ``{status, content}`` envelope); used by
        dataset recording and camera warm-up.

        Returns:
            ``(rgb, depth, meta)``. On success ``rgb`` is a uint8
            ``(H, W, 3)`` array, ``depth`` a float32 ``(H, W)`` array, and
            ``meta`` carries ``text`` plus (RTX path) a ``json`` sub-dict.
            On failure ``rgb`` / ``depth`` are ``None`` and
            ``meta["error"]`` holds the human-readable reason.
        """
        with self._lock:
            if not self._world_created:
                return None, None, {"error": "No world created."}

            w = width or self._config.camera_width
            h = height or self._config.camera_height

            if self._config.render_mode == "headless":
                # Return blank frames in headless mode. Most CI flows
                # land here; Isaac's RTX path-tracer is unavailable.
                return (
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    {"text": f"Rendered (headless, no RTX): {w}x{h}"},
                )

            if camera_name not in self._cameras:
                # No camera configured - return blank. Caller probably
                # forgot to call add_camera or typo'd the name.
                return (
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    {"text": f"Rendered (no camera): {w}x{h}"},
                )

            cam = self._cameras[camera_name]

            if cam.handle is None:
                # Defensive fallback: blank frames sized to the camera's
                # registered resolution (what the caller asked for at
                # add_camera) rather than this method's width/height.
                return (
                    np.zeros((cam.height, cam.width, 3), dtype=np.uint8),
                    np.zeros((cam.height, cam.width), dtype=np.float32),
                    {"text": f"Rendered (Phase-1 camera, no RTX handle): {cam.width}x{cam.height}"},
                )

            # RTX path: pull real frames from the Camera handle.
            try:
                rgba = cam.handle.get_rgba()
                # ``get_rgba`` returns ``(H, W, 4)`` or ``(H, W, 3)``
                # depending on the build; a not-yet-warmed RTX render
                # product returns a malformed / empty buffer (0-D, 1-D or
                # 0-size). Validate the shape BEFORE slicing -- a 0-D
                # array makes ``[..., :3]`` raise IndexError -- so the
                # not-ready case surfaces as a structured RuntimeError
                # (caught below) and the warm-up loop can retry.
                arr = np.asarray(rgba)
                if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
                    raise RuntimeError(
                        f"camera {camera_name!r} returned a malformed RGB buffer "
                        f"(shape {arr.shape}); the RTX render product "
                        "likely hasn't accumulated a frame yet -- step the world a "
                        "few times after add_camera before rendering."
                    )
                # Slice to RGB defensively so the returned shape is stable
                # for downstream agents.
                rgb = arr[..., :3]
                depth_raw = cam.handle.get_depth()
                if depth_raw is None:
                    # Depth annotator not enabled (rgba is on by default
                    # but depth is opt-in). Surface a zero-depth array
                    # sized to rgb so callers see a stable shape, plus a
                    # WARNING so misconfigured cameras are visible.
                    logger.warning(
                        "Camera '%s': get_depth() returned None (depth annotator not enabled). "
                        "Returning zero-depth array; "
                        "check add_distance_to_image_plane_to_frame() in add_camera.",
                        camera_name,
                    )
                    depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                else:
                    depth = np.asarray(depth_raw)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, IndexError) as e:
                # Cleanup-clause shape mirrors create_world. IndexError is
                # included so a 0-D ``get_rgba`` buffer during RTX warm-up
                # surfaces here even if the pre-slice guard is bypassed.
                logger.error("Failed to render camera '%s': %s", camera_name, e)
                return None, None, {"error": f"Failed to render camera '{camera_name}': {e}"}

            render_info = {
                "rtx": True,
                "prim_path": cam.prim_path,
                "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
                "render_mode": self._config.render_mode,
            }
            return (
                rgb,
                depth,
                {
                    "text": f"Rendered (RTX {self._config.render_mode}): {cam.width}x{cam.height}",
                    "json": render_info,
                },
            )

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render a camera to raw ``(rgb, depth)`` ndarrays (metric depth).

        Public counterpart of the internal :meth:`_render_frame` for
        in-process consumers such as
        :class:`strands_robots.rendering.HybridCompositor`. Unlike
        :meth:`_render_frame`, this method **raises** on every degraded
        path (headless mode, unknown camera, camera without an RTX
        handle), so a compositing consumer can never silently receive
        black pixels with zero depth. Isaac's depth annotator reports
        pixels with no geometry as ``0`` or non-finite; treat both
        extremes as background.

        Concurrency: takes ``self._lock``; rendering must be driven from
        the thread that owns the ``SimulationApp`` (use
        :meth:`run_on_main` from worker threads).

        Args:
            camera_name: a camera previously added via ``add_camera``.
            width: must be ``None`` or the camera's native render width
                (fixed at ``add_camera`` time); a mismatch raises.
            height: same contract as ``width``.

        Returns:
            ``(rgb, depth)`` -- ``(H, W, 3) uint8`` and ``(H, W) float32``.

        Raises:
            RuntimeError: no world, headless render mode, camera without
                an RTX handle, or an RTX render failure.
            KeyError: unknown camera name.
            ValueError: ``width``/``height`` differ from the camera's
                native render resolution.
        """
        with self._lock:
            if not self._world_created:
                raise RuntimeError("No world created. Call create_world first.")
            if self._config.render_mode == "headless":
                raise RuntimeError(
                    "get_frame is unavailable in headless render mode (no RTX frames are produced); "
                    "use render_mode='rtx_realtime' or consume the envelope render() fallback."
                )
            if camera_name not in self._cameras:
                raise KeyError(f"Camera '{camera_name}' not found. Available: {sorted(self._cameras)}")
            cam = self._cameras[camera_name]
            if cam.handle is None:
                raise RuntimeError(f"Camera '{camera_name}' has no live RTX handle; re-add it via add_camera().")
            for arg_name, arg, native in (("width", width, cam.width), ("height", height, cam.height)):
                if arg is not None and int(arg) != int(native):
                    raise ValueError(
                        f"Isaac cameras render at the resolution fixed at add_camera time; "
                        f"requested {arg_name}={arg} but camera '{camera_name}' renders at "
                        f"{cam.width}x{cam.height}. Re-add the camera with the desired size."
                    )
            rgb, depth, meta = self._render_frame(camera_name)
        if rgb is None:
            raise RuntimeError(str(meta.get("error", f"Failed to render camera '{camera_name}'")))
        depth_arr = None if depth is None else np.asarray(depth, dtype=np.float32)
        return np.asarray(rgb, dtype=np.uint8), depth_arr

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return pinhole :class:`~strands_robots.rendering.CameraParams`.

        Intrinsics come from ``Camera.get_intrinsics_matrix()``, the pose
        from ``Camera.get_world_pose()``. The USD camera prim's local axes
        are offset from the OpenGL optical frame ``CameraParams`` promises
        (+X right, +Y up, -Z forward): prim +X -> GL -Z, prim +Y -> GL -X,
        prim +Z -> GL +Y. That fixed correction
        (``R_gl = R_prim @ PRIM_TO_GL``) is applied here so a composited
        background is upright and aligned with the RTX foreground.

        Args:
            camera_name: a camera previously added via ``add_camera``.
            width: must be ``None`` or the camera's native render width
                (intrinsics are only valid at native resolution).
            height: same contract as ``width``.

        Raises:
            RuntimeError: no world, or the camera has no live RTX handle.
            KeyError: unknown camera name.
            ValueError: ``width``/``height`` differ from the native
                render resolution.
        """
        from strands_robots.rendering import CameraParams

        with self._lock:
            if not self._world_created:
                raise RuntimeError("No world created. Call create_world first.")
            if camera_name not in self._cameras:
                raise KeyError(f"Camera '{camera_name}' not found. Available: {sorted(self._cameras)}")
            cam = self._cameras[camera_name]
            if cam.handle is None:
                raise RuntimeError(
                    f"Camera '{camera_name}' has no live RTX handle -- intrinsics/pose cannot be "
                    "read off a registration-only camera. Re-add it via add_camera()."
                )
            for arg_name, arg, native in (("width", width, cam.width), ("height", height, cam.height)):
                if arg is not None and int(arg) != int(native):
                    raise ValueError(
                        f"Isaac camera intrinsics are only valid at the native render resolution; "
                        f"requested {arg_name}={arg} but camera '{camera_name}' renders at "
                        f"{cam.width}x{cam.height}. Re-add the camera with the desired size."
                    )
            K = np.asarray(cam.handle.get_intrinsics_matrix(), dtype=np.float64).reshape(3, 3)
            position, quat_wxyz = cam.handle.get_world_pose()
            w_px, h_px = int(cam.width), int(cam.height)

        position = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        # Fixed camera-local correction, USD camera prim basis -> OpenGL
        # optical frame (see docstring). R_gl = R_prim @ PRIM_TO_GL.
        prim_to_gl = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz) @ prim_to_gl
        T[:3, 3] = position
        # Isaac exposes no scene-level clip planes on the handle; carry the
        # conventional near plane and a far plane "at infinity" for the
        # compositor's depth convention.
        return CameraParams(K=K, T_world_cam=T, width=w_px, height=h_px, znear=0.01, zfar=1_000_000.0)

    def _warmup_camera(self, name: str, n_steps: int) -> bool:
        """Step the world (with rendering) until camera ``name`` yields a frame.

        Isaac's RTX render product accumulates no frame until the world is
        stepped with rendering enabled, so a fresh camera returns a
        malformed / empty ``get_rgba()`` buffer at first. Steps up to
        ``n_steps`` times, returning ``True`` as soon as :meth:`render`
        reports success and ``False`` if the camera never warmed up
        (logged at WARNING). Never raises: a step / render failure ends
        the loop, and render()'s own 0-D guard still covers a
        not-yet-ready product.
        """
        if self._world is None:
            return False
        for i in range(max(1, n_steps)):
            try:
                self._world.step(render=True)
                self._sim_time += self._config.physics_dt
                self._step_count += 1
                if self.render(camera_name=name).get("status") == "success":
                    logger.debug("Camera %r warmed up after %d step(s)", name, i + 1)
                    return True
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, IndexError) as e:
                # Stepping / rendering a partially-initialised stage can
                # raise on surface drift; warm-up is best-effort, so log
                # and stop rather than failing the already-registered
                # camera. Programming bugs (NameError) still propagate.
                logger.debug("Camera %r warm-up step %d failed: %s", name, i + 1, e)
                break
        logger.warning(
            "Camera %r did not produce a valid frame after %d warm-up step(s); "
            "the first render() may return an error until the RTX product accumulates a frame.",
            name,
            n_steps,
        )
        return False

    def add_camera(
        self,
        name: str = "default",
        position: list[float] | None = None,
        target: list[float] | None = None,
        width: int | None = None,
        height: int | None = None,
        fov: float = 60.0,
    ) -> dict[str, Any]:
        """Add an RTX camera to the scene.

        Instantiates the USD camera prim and stores the handle on the
        ``_CameraState`` for later retrieval by :meth:`render`.

        Parameters
        ----------
        name : str
            Camera identifier. Default ``"default"``. Must be unique; a
            duplicate is rejected with a structured error envelope.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[2.0, 2.0, 2.0]``.
        target : list[float], optional
            World-space look-at point ``[x, y, z]``. If ``None``, the
            camera keeps its constructed (identity) orientation.
        width : int, optional
            Image width in pixels. Defaults to ``IsaacConfig.camera_width``.
        height : int, optional
            Image height in pixels. Defaults to ``IsaacConfig.camera_height``.
        fov : float
            Horizontal field of view in degrees. Default 60.0. Mapped
            onto ``Camera.set_focal_length`` via the pinhole relation
            ``focal_length = horizontal_aperture / (2 * tan(fov/2))``.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope; ``json`` carries the resolved ``prim_path``,
            ``position``, ``target``, ``resolution``, ``fov``, and the
            computed ``focal_length``.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if name in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' already exists."}],
                }

            w = int(width or self._config.camera_width)
            h = int(height or self._config.camera_height)
            pos = list(position) if position is not None else [2.0, 2.0, 2.0]
            tgt = list(target) if target is not None else None
            fov_deg = float(fov)

            # Render at a higher NATIVE resolution if the requested output
            # is small so the DLSS upscaler stays above its temporal-ghost
            # threshold (see ``_MIN_RENDER_PX``); frames are downscaled
            # back to ``(w, h)`` before return. Skipped in headless mode.
            out_w, out_h = w, h
            if self._config.render_mode != "headless" and w < _MIN_RENDER_PX:
                scale = _MIN_RENDER_PX / float(w)
                w = _MIN_RENDER_PX
                h = int(round(h * scale))

            prim_path = f"{self._config.stage_path}/Cameras/{name}"

            try:
                handle, focal_length_mm = self._create_camera_prim(
                    name=name,
                    prim_path=prim_path,
                    position=pos,
                    target=tgt,
                    width=w,
                    height=h,
                    fov_deg=fov_deg,
                )
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Either the camera is constructed and registries update,
                # or it fails with a structured envelope and neither does.
                logger.error("Failed to add camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add camera '{name}': {e}"}],
                }

            self._prim_registry.append(prim_path)
            cam_state = _CameraState(name=name, prim_path=prim_path, width=w, height=h)
            cam_state.handle = handle
            self._cameras[name] = cam_state
            # Track requested OUTPUT size (may differ from native render
            # size when DLSS upscaling required a larger native frame).
            self._cam_out_size[name] = (out_w, out_h)

            # Warm up the RTX render product so the first render() /
            # recording call sees a valid frame. Skipped in headless mode;
            # best-effort (a warm-up failure never fails add_camera).
            if self._config.render_mode != "headless" and self._camera_warmup_steps > 0:
                self._warmup_camera(name, self._camera_warmup_steps)

            cam_info = {
                "name": name,
                "prim_path": prim_path,
                "position": pos,
                "target": tgt,
                "resolution": [w, h],
                "fov": fov_deg,
                "focal_length_mm": focal_length_mm,
            }
            logger.info(
                "Added camera '%s' at pos=%s target=%s res=%dx%d fov=%.1f",
                name,
                pos,
                tgt,
                w,
                h,
                fov_deg,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (f"Camera '{name}' added at {pos}, resolution={w}x{h}, fov={fov_deg}"),
                        "json": cam_info,
                    }
                ],
            }

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a camera from the scene.

        Deletes the underlying USD camera prim and prunes the in-Python
        registries.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}]}`` envelope.
            Returns ``error`` if the camera is unknown to ``_cameras``.
        """
        with self._lock:
            if name not in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' not found."}],
                }

            prim_path = self._cameras[name].prim_path

            # Cameras are standalone USD prims (not scene objects), so
            # removal goes via the stage utility. Same except tuple as
            # add_camera: a transient stage error returns the structured
            # envelope and leaves bookkeeping intact for retry.
            try:
                if self._world is not None:
                    try:
                        from isaacsim.core.utils.prims import (  # type: ignore[import-not-found]
                            delete_prim,
                        )
                    except ImportError:
                        from omni.isaac.core.utils.prims import (  # type: ignore[import-not-found]
                            delete_prim,
                        )

                    delete_prim(prim_path)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                logger.error("Failed to remove camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove camera '{name}': {e}"}],
                }

            del self._cameras[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed camera '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Camera '{name}' removed."}],
            }

    # --- Recording (rollout video) -----------------------------------------
    #
    # MuJoCo records rollout videos from a daemon thread; Isaac can't (the
    # RTX renderer + ``Camera.get_rgba`` are bound to the thread that
    # booted ``SimulationApp``). So the Isaac recorder is *synchronous*:
    # it returns an ``on_frame`` closure that the eval driver wires into
    # ``evaluate_benchmark(..., on_frame=...)``; the closure captures one
    # frame per applied control step, and ``stop_cameras_recording``
    # flushes the buffers to ``{name}__{camera}.mp4`` -- the same filename
    # convention MuJoCo uses, so cross-backend video discovery picks up
    # Isaac rows uniformly.

    def start_cameras_recording(
        self,
        cameras: list[str] | None = None,
        output_dir: str | None = None,
        fps: int = 30,
        name: str | None = None,
        max_frames_per_camera: int = 3000,
    ) -> dict[str, Any]:
        """Begin a synchronous rollout-video recording.

        Sets up one in-memory RGB buffer per camera and returns an
        ``on_frame(step, observation, action)`` closure in the result's
        ``json`` block. Wire it into :meth:`evaluate_benchmark`'s
        ``on_frame=`` kwarg; it captures one frame per applied control
        step on the eval thread (no daemon thread -- Isaac's RTX renderer
        is thread-bound). :meth:`stop_cameras_recording` then flushes the
        buffers to ``{name}__{camera}.mp4`` under ``output_dir``
        (MuJoCo's filename convention).

        Parameters
        ----------
        cameras : list[str], optional
            Camera names to record. ``None`` = every registered camera.
            Unknown names error loudly.
        output_dir : str, optional
            Defaults to ``$TMPDIR/strands_robots/recordings``.
        fps : int
            Encoded MP4 frame rate. Default 30. Must be a positive whole
            number - the rate the flush encodes at, refused here rather
            than after a rollout's frames have been buffered.
        name : str
            Filename tag. Auto-generated (``rec_<uuid>``) when ``None``.
        max_frames_per_camera : int
            Safety cap on in-memory buffers. Frames beyond the cap are
            silently dropped. Default 3000. Must be a positive whole
            number; a cap below 1 drops every frame.

        Returns
        -------
        dict
            On success: ``{"status": "success", "content": [{"text": ...},
            {"json": {"on_frame": <callable>, "cameras": [...],
            "output_dir": ..., "name": ...}}]}``. The ``on_frame`` closure
            isn't JSON-serializable; Python callers unpack it from the
            json block. On error (unusable ``fps`` or
            ``max_frames_per_camera``, no world, already recording,
            unknown cameras, none to record):
            ``{"status": "error", ...}``.
        """
        import os as _os
        import tempfile as _tempfile
        import time as _time
        import uuid as _uuid

        # Refuse a frame count the recorder cannot honor before any filesystem
        # or buffer work: ``fps`` reaches ``encode_clip`` at flush time, which
        # refuses a rate it cannot encode at, and a non-positive frame cap
        # drops every captured frame.
        if error := _cameras_recording_option_error("start_cameras_recording", fps, max_frames_per_camera):
            return error

        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}

            rec_state = self._cams_rec_state
            if rec_state and rec_state.get("running"):
                cur = rec_state["name"]
                return {
                    "status": "error",
                    "content": [{"text": f"Already recording '{cur}'. Call stop_cameras_recording() first."}],
                }

            if cameras is None:
                names = list(self._cameras.keys())
            else:
                unresolved = [c for c in cameras if c not in self._cameras]
                if unresolved:
                    return {
                        "status": "error",
                        "content": [
                            {"text": (f"Camera(s) not found: {unresolved}. Available: {list(self._cameras.keys())}")}
                        ],
                    }
                names = list(cameras)
            if not names:
                return {"status": "error", "content": [{"text": "No cameras to record."}]}

            out_dir = _os.path.abspath(
                output_dir or _os.path.join(_tempfile.gettempdir(), "strands_robots", "recordings")
            )
            _os.makedirs(out_dir, exist_ok=True)
            tag = name or f"rec_{_uuid.uuid4().hex[:8]}"

            buffers: dict[str, list] = {cam: [] for cam in names}
            paths = {cam: _os.path.join(out_dir, f"{tag}__{cam}.mp4") for cam in names}

            state: dict[str, Any] = {
                "running": True,
                "name": tag,
                "cameras": names,
                "fps": fps,
                "buffers": buffers,
                "paths": paths,
                "errors": dict.fromkeys(names, 0),
                "output_dir": out_dir,
                "started_at": _time.time(),
                "max_frames": max_frames_per_camera,
            }
            self._cams_rec_state = state

        def on_frame(step: int, observation: dict, action: dict) -> None:
            """Capture one RGB frame per camera (runs on the eval thread).

            Best-effort: a render failure on a single camera/step
            increments that camera's error counter rather than raising,
            so a transient RTX hiccup doesn't abort the whole eval.
            """
            st = getattr(self, "_cams_rec_state", None)
            if not st or not st.get("running"):
                return
            for cam in st["cameras"]:
                if len(st["buffers"][cam]) >= st["max_frames"]:
                    continue
                try:
                    rgb, _depth, _meta = self._render_frame(camera_name=cam)
                    if rgb is None:
                        st["errors"][cam] += 1
                        continue
                    arr = np.asarray(rgb)
                    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
                        st["errors"][cam] += 1
                        continue
                    st["buffers"][cam].append(np.ascontiguousarray(arr[..., :3].astype(np.uint8)))
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError):
                    st["errors"][cam] += 1

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Recording '{tag}' armed for cameras {names}. "
                        "Pass the returned on_frame to evaluate_benchmark(on_frame=...), "
                        "then call stop_cameras_recording()."
                    ),
                    "json": {
                        "on_frame": on_frame,
                        "cameras": names,
                        "output_dir": out_dir,
                        "name": tag,
                        "paths": paths,
                    },
                }
            ],
        }

    def stop_cameras_recording(self) -> dict[str, Any]:
        """Stop recording and flush captured frames to MP4.

        Encodes each camera's in-memory RGB buffer to
        ``{name}__{camera}.mp4`` under the ``output_dir`` passed to
        :meth:`start_cameras_recording`, using ``imageio`` (the same
        encoder the MuJoCo recorder uses). Idempotent: a no-op success
        when nothing is recording.

        Best-effort: per-camera flush failures are reported in the result
        (``frames`` / ``errors`` / ``size_kb``) but never raise, so a
        partial encode still yields a structured success response.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}, {"json"}]}``
            envelope. ``json`` carries ``recording`` (the tag) and an
            ``artifacts`` list of ``{camera, path, frames, errors,
            size_kb}`` per camera.
        """
        import os as _os
        import time as _time

        with self._lock:
            state = getattr(self, "_cams_rec_state", None)
            if not state or not state.get("running"):
                return {"status": "success", "content": [{"text": "Was not recording cameras."}]}
            state["running"] = False
            self._cams_rec_state = None

        from strands_robots.rendering.video import encode_clip

        elapsed = _time.time() - state["started_at"]
        lines = [
            f"Stopped '{state['name']}' after {elapsed:.1f}s",
            f"   output_dir: {state['output_dir']}",
        ]
        artifacts = []
        for cam in state["cameras"]:
            frames_buffer = state["buffers"][cam]
            path = state["paths"][cam]
            errors = state["errors"][cam]
            frames_written = 0
            size_kb = 0.0
            flush_error = None
            if frames_buffer:
                # Shared encoder (strands_robots.rendering.video).
                try:
                    encode_clip(frames_buffer, path, fps=state["fps"])
                    frames_written = len(frames_buffer)
                except ImportError:
                    return {
                        "status": "error",
                        "content": [{"text": "imageio not installed. pip install imageio imageio-ffmpeg"}],
                    }
                except (RuntimeError, ValueError) as e:
                    # ``encode_clip`` refused the clip: ``RuntimeError`` when it
                    # wrote no file, ``ValueError`` when it will not encode at
                    # the requested rate. ``start_cameras_recording`` pre-flights
                    # that rate, so the second is unreachable through the tool
                    # pair; it is caught anyway because this method's contract is
                    # best-effort and never-raise, and the flush is the last
                    # chance to hand back the buffered frames' fate. Report the
                    # reason on the artifact line and keep ``frames_written`` at
                    # 0 rather than claiming frames that reached no file.
                    flush_error = f"{type(e).__name__}: {e}"
                    logger.warning("camera recorder flush failed for %r -> %s: %s", cam, path, flush_error)
                if _os.path.exists(path):
                    size_kb = _os.path.getsize(path) / 1024
            line = (
                f"   {cam:20s} {frames_written:>5d} frames  {size_kb:>7.1f} KB  "
                f"({errors} errors)  -> {_os.path.basename(path)}"
            )
            if flush_error:
                line += f"  [flush failed: {flush_error}]"
            lines.append(line)
            artifact = {
                "camera": cam,
                "path": path,
                "frames": frames_written,
                "errors": errors,
                "size_kb": size_kb,
            }
            if flush_error:
                artifact["flush_error"] = flush_error
            artifacts.append(artifact)

        return {
            "status": "success",
            "content": [
                {"text": "\n".join(lines)},
                {"json": {"recording": state["name"], "artifacts": artifacts}},
            ],
        }

    def _create_camera_prim(
        self,
        *,
        name: str,
        prim_path: str,
        position: list[float],
        target: list[float] | None,
        width: int,
        height: int,
        fov_deg: float,
    ) -> tuple[Any, float]:
        """Construct the Isaac camera prim + apply look-at + FOV.

        Returns the camera handle plus the resolved focal length in mm so
        :meth:`add_camera` can surface it. Lazy-imports the ``Camera``
        sensor and ``set_camera_view`` (modern path first, legacy 4.x
        fallback) so the module loads without Isaac Sim installed.
        """
        import math

        import numpy as np  # type: ignore[import-not-found]

        try:
            from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
        except ImportError:
            from omni.isaac.sensor import Camera  # type: ignore[import-not-found]

        camera = Camera(
            prim_path=prim_path,
            name=name,
            position=np.asarray(position, dtype=float),
            resolution=(int(width), int(height)),
        )
        # ``initialize`` allocates the RTX render product + annotators.
        # Some Camera builds defer this to first ``get_rgba()`` call;
        # call it explicitly so an init-time failure surfaces here
        # (and gets caught by the cleanup clause in add_camera) rather
        # than silently on the first render attempt.
        camera.initialize()

        # Map FOV (deg, horizontal) to focal length (mm) via the pinhole
        # relation focal_length = horizontal_aperture / (2 * tan(fov/2)).
        # The aperture MUST be read back from the prim (assuming a nominal
        # 24 mm yields a telephoto instead of the intended FOV on Isaac's
        # Camera); deriving from the read-back aperture makes the pixel
        # intrinsics fx = width / (2*tan(fov/2)) exactly, independent of
        # the aperture's absolute value or units.
        try:
            horizontal_aperture_mm = float(camera.get_horizontal_aperture())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            horizontal_aperture_mm = 24.0
        focal_length_mm = horizontal_aperture_mm / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        camera.set_focal_length(focal_length_mm)

        # Enable the depth annotator (rgba is on by default, depth is
        # opt-in; without it ``camera.get_depth()`` returns ``None``).
        # Idempotent on repeat calls.
        try:
            camera.add_distance_to_image_plane_to_frame()
        except (AttributeError, RuntimeError):
            # Older builds expose this as ``add_depth_to_frame``; if that
            # also fails, render()'s defensive None-handling covers it.
            try:
                camera.add_depth_to_frame()
            except (AttributeError, RuntimeError):
                logger.debug(
                    "Camera %s: depth annotator not enabled; ``get_depth()`` will return None",
                    name,
                )

        # Apply look-at after focal-length so the camera's forward axis
        # is correctly oriented at the target. ``set_camera_view`` works
        # on any USD camera prim by path; no Camera-specific API.
        if target is not None:
            try:
                from isaacsim.core.utils.viewports import (  # type: ignore[import-not-found]
                    set_camera_view,
                )
            except ImportError:
                from omni.isaac.core.utils.viewports import (  # type: ignore[import-not-found]
                    set_camera_view,
                )

            set_camera_view(eye=position, target=target, camera_prim_path=prim_path)

        return camera, focal_length_mm

    # --- Isaac-specific: Fleet Replication -----------------------------------

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the current scene into parallel environments.

        Uses ``omni.isaac.cloner.Cloner`` for GPU-efficient replication.

        Parameters
        ----------
        num_envs : int, optional
            Number of environments. Defaults to config.num_envs.

        Returns
        -------
        dict
            Status dict with replication info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._robots:
                return {
                    "status": "error",
                    "content": [{"text": "Add at least one robot first."}],
                }

            n = num_envs or self._config.num_envs

            t0 = time.perf_counter()
            # In full implementation: use omni.isaac.cloner.Cloner
            # to replicate the scene N times
            self._replicated = True
            self._num_envs_active = n
            elapsed = time.perf_counter() - t0

            logger.info("Replicated to %d envs in %.2fs", n, elapsed)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Replicated to {n} environments. "
                            f"Build time: {elapsed * 1000:.0f}ms. "
                            f"Device: {self._config.device}."
                        ),
                        "json": {
                            "num_envs": n,
                            "build_time_ms": elapsed * 1000,
                        },
                    }
                ],
            }

    # --- Private Implementation ----------------------------------------------

    def _load_usd_robot(self, prim_path: str, usd_path: str, position: list[float]) -> tuple[list[str], Any]:
        """Load a robot from a USD file. Returns ``(joint_names, articulation)``.

        References the USD into the stage at ``prim_path``, wraps the prim
        in an Articulation, calls ``initialize()`` (without which
        ``dof_names`` is ``None`` on most builds), applies the requested
        ``position`` (identity ``[0, 0, 0]`` skipped), and returns the
        joint names alongside the live handle.

        Raises propagate -- the caller (``add_robot`` USD branch) wraps
        this in the standard cleanup-clause tuple so Isaac-side surface
        drift returns a structured error envelope.
        """
        import numpy as np  # type: ignore[import-not-found]

        Articulation = _import_articulation_cls()  # noqa: N806

        try:
            from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )
        except ImportError:
            from omni.isaac.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )

        # Stage reference: the USD's default prim becomes a child of
        # ``prim_path``; subsequent Articulation lookups walk that path.
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # Wrap + initialise. The articulation name must be unique within
        # the scene registry; the leaf of ``prim_path`` is the
        # caller-visible robot name by construction.
        articulation_name = prim_path.rsplit("/", 1)[-1]
        articulation = Articulation(prim_path=prim_path, name=articulation_name)
        articulation.initialize()
        # ``add_reference_to_stage`` honours ``prim_path``; record it as
        # the actual landing path (symmetry with the URDF branch) so
        # ``add_robot`` can seed ``_RobotState.actual_prim_path``.
        try:
            articulation._strands_actual_prim_path = prim_path  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

        # Only call set_world_pose for a non-default placement (saves a
        # tensor round-trip on the common ``position=[0, 0, 0]`` case).
        if position is not None and any(p != 0.0 for p in position):
            articulation.set_world_pose(position=np.asarray(position, dtype=float))

        # Coerce ``dof_names=None`` (no Articulation root on the prim) to
        # ``[]`` so callers see the documented empty-joint-list mode.
        joint_names = list(articulation.dof_names) if articulation.dof_names else []

        logger.info(
            "Loaded USD robot at %s from %s (%d joints, articulation=initialized)",
            prim_path,
            usd_path,
            len(joint_names),
        )
        return joint_names, articulation

    def _load_urdf_robot(self, prim_path: str, urdf_path: str, position: list[float]) -> tuple[list[str], Any]:
        """Load a robot from a URDF file. Returns ``(joint_names, articulation)``.

        Runs the URDF importer (fixed-base manipulator defaults), honours
        the importer's actual landing prim path (it may differ from the
        requested one), wraps + initialises an Articulation, applies the
        requested ``position`` (origin skipped), and returns joint names
        (``dof_names=None`` coerced to ``[]``) alongside the handle --
        same shape as :meth:`_load_usd_robot`.

        Raises propagate; the caller (``add_robot`` URDF branch) wraps in
        the standard cleanup-clause tuple.
        """
        import numpy as np  # type: ignore[import-not-found]

        Articulation = _import_articulation_cls()  # noqa: N806

        # Isaac Sim's URDF importer API varies across releases:
        # * 6.0 exposes high-level ``URDFImporter`` + ``URDFImporterConfig``
        #   classes (the ``_urdf`` C-binding is no longer importable).
        # * 4.5/5.x used ``isaacsim.asset.importer.urdf._urdf`` with
        #   ``acquire_urdf_interface().parse_urdf()/import_robot()``.
        # * pre-4.5 used ``omni.importer.urdf._urdf``.
        # Try the modern 6.0 class API first, then the legacy ``_urdf`` ifaces.
        import os

        urdf_root, urdf_filename = os.path.split(os.path.abspath(urdf_path))
        imported_prim_path = None

        URDFImporter = URDFImporterConfig = None  # noqa: N806
        try:
            from isaacsim.asset.importer.urdf import (  # type: ignore[import-not-found,no-redef]
                URDFImporter,
                URDFImporterConfig,
            )
        except ImportError:
            URDFImporter = URDFImporterConfig = None  # noqa: N806

        if URDFImporter is not None and URDFImporterConfig is not None:
            # Isaac Sim 6.0 high-level API. Fixed base, no fixed-joint
            # merge (keeps joint names stable), self-collision off, drive
            # type 'position' so the articulation can be position-commanded.
            cfg = URDFImporterConfig()
            cfg.urdf_path = os.path.abspath(urdf_path)
            for attr, val in (
                ("fix_base", True),
                ("merge_fixed_joints", False),
                ("allow_self_collision", False),
                ("collision_from_visuals", False),
            ):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, val)
            if hasattr(cfg, "joint_drive_type"):
                try:
                    cfg.joint_drive_type = "position"
                except (AttributeError, TypeError, ValueError):  # enum vs str varies; leave default
                    pass
            # Strong position-drive gains so the arm holds against gravity
            # (a URDF with no <dynamics> drive params gets gains too soft
            # to track commanded targets).
            for attr, val in (("override_joint_stiffness", 1.0e5), ("override_joint_damping", 1.0e4)):
                if hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, val)
                    except (AttributeError, TypeError, ValueError):
                        pass
            importer = URDFImporter(config=cfg) if _accepts_config_kw(URDFImporter) else URDFImporter()
            if hasattr(importer, "config"):
                try:
                    importer.config = cfg
                except (AttributeError, TypeError):
                    pass
            # Isaac Sim 6.0 ``import_urdf()`` converts URDF -> USD on disk and
            # returns the USD path (NOT a live-stage prim path). Reference that
            # USD onto the live stage at our prim_path, then wrap that prim as an
            # Articulation.
            usd_out = importer.import_urdf()
            if not isinstance(usd_out, str) or not usd_out:
                raise RuntimeError(f"URDF import (6.0 API) returned no USD path for {urdf_path!r}")
            from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]

            add_reference_to_stage(usd_path=usd_out, prim_path=prim_path)
            imported_prim_path = prim_path
        else:
            try:
                from isaacsim.asset.importer.urdf import _urdf  # type: ignore[import-not-found]
            except ImportError:
                from omni.importer.urdf import _urdf  # type: ignore[import-not-found]

            import_config = _urdf.ImportConfig()
            import_config.fix_base = True
            import_config.import_inertia_tensor = True
            import_config.create_physics_scene = False
            import_config.distance_scale = 1.0
            if hasattr(import_config, "merge_fixed_joints"):
                import_config.merge_fixed_joints = False
            if hasattr(import_config, "self_collision"):
                import_config.self_collision = False
            if hasattr(import_config, "make_default_prim"):
                import_config.make_default_prim = False

            urdf_iface = _urdf.acquire_urdf_interface()
            urdf_robot = urdf_iface.parse_urdf(urdf_root, urdf_filename, import_config)
            if urdf_robot is None:
                raise RuntimeError(f"URDF parse failed for {urdf_path!r}")
            imported_prim_path = urdf_iface.import_robot(urdf_root, urdf_filename, urdf_robot, import_config, "")
            if not imported_prim_path:
                raise RuntimeError(f"URDF import failed for {urdf_path!r} via _urdf.import_robot")

        # Use the importer's actual landing path for Articulation
        # construction; caller bookkeeping records it for later lookups.
        actual_prim_path = imported_prim_path

        # Articulation wrap + initialise.
        articulation_name = actual_prim_path.rsplit("/", 1)[-1]
        articulation = Articulation(prim_path=actual_prim_path, name=articulation_name)
        articulation.initialize()
        # Strong position-drive PD gains so the arm holds against gravity
        # and tracks commanded targets (see the importer-config note).
        try:
            ndof = len(articulation.dof_names) if articulation.dof_names else 0
            if ndof:
                kp = np.full(ndof, 1.0e5, dtype=float)
                kd = np.full(ndof, 1.0e4, dtype=float)
                set_gains = getattr(articulation, "set_gains", None)
                if callable(set_gains):
                    set_gains(kps=kp, kds=kd)
                else:
                    ctrl = getattr(articulation, "get_articulation_controller", None)
                    if callable(ctrl):
                        ctrl().set_gains(kps=kp, kds=kd)
                # Raise the per-joint max effort far above the SO-101
                # URDF's tiny ``effort=10`` limit: with it, PhysX clamps
                # the gripper drive torque regardless of PD stiffness and
                # the jaw can't hold a grasp. 1000 gives real clamping
                # authority while staying physical.
                set_max = getattr(articulation, "set_max_efforts", None)
                if callable(set_max):
                    try:
                        set_max(np.full(ndof, 1.0e3, dtype=float))
                    except (RuntimeError, ValueError, TypeError, IndexError):
                        # Some builds expect a (M, K) batch for the view.
                        set_max(np.full((1, ndof), 1.0e3, dtype=float))
        except (AttributeError, TypeError, ValueError, RuntimeError, IndexError):  # gain set is best-effort
            logger.debug("set drive gains failed (non-fatal)", exc_info=True)
        # Stash the importer's actual landing path as a sidecar attribute
        # so ``add_robot`` can record it on ``_RobotState.actual_prim_path``
        # for later USD-stage walks (e.g. ``gripper_frame_pose``).
        try:
            articulation._strands_actual_prim_path = actual_prim_path  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Some Articulation builds don't allow attribute assignment;
            # caller falls back to the requested prim_path in that case.
            pass

        # Position. Same skip-origin shortcut as ``_load_usd_robot``.
        if position is not None and any(p != 0.0 for p in position):
            articulation.set_world_pose(position=np.asarray(position, dtype=float))

        joint_names = list(articulation.dof_names) if articulation.dof_names else []

        logger.info(
            "Loaded URDF robot at %s from %s (%d joints, articulation=initialized)",
            actual_prim_path,
            urdf_path,
            len(joint_names),
        )
        return joint_names, articulation

    # --- SimEngine: extra helpers for the SO-101 cuRobo example -------------
    #
    # Migrated in from an example-local Isaac adapter. Three concerns the
    # headless ``SimEngine`` core doesn't cover:
    #
    # 1. Main-thread pump (``pump`` / ``run_pump_forever`` / ``run_on_main``)
    #    -- Isaac's renderer + physics may only be driven from the thread
    #    that created ``SimulationApp``; the pump runs there and is the
    #    single place that advances the sim and renders the cameras.
    # 2. Kinematic teleport-grasp helpers (``set_object_collision``,
    #    ``gripper_frame_pos``/``gripper_frame_pose``, ``move_object``) --
    #    the actuator-less SO-101 URDF can't grip via friction, so the
    #    collector teleport-follows the cube to the gripper.
    # 3. DLSS ghost mitigation (``_converge_render``, ``_resize_rgb``,
    #    ``_configure_renderer``, ``_add_lighting``, plus the
    #    ``add_camera`` native-resolution upscale) -- render wide and hold
    #    the pose static for a few converge ticks so frames stay crisp.
    #
    # The headless / CI path doesn't engage any of these.

    # --- main-thread pump --------------------------------------------------

    def pump(self, render: bool = True) -> None:
        """Drain queued actions, step once, refresh caches. MAIN THREAD ONLY.

        A web UI calls ``get_observation``/``send_action`` from worker
        threads where Isaac's renderer / physics deadlock. Those calls
        instead enqueue actions and read cached frames; this pump (run
        on the owning main thread) is the single place that actually
        advances the sim and renders the cameras.
        """
        if not self._world_created or self._world is None:
            return
        # 1. Apply any actions queued by worker threads, counting them.
        n_actions = 0
        while not self._action_q.empty():
            try:
                fn = self._action_q.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
                n_actions += 1
            except (RuntimeError, ValueError, AttributeError, TypeError, KeyError, IndexError):
                # Queued worker actions are best-effort; narrow to what
                # Isaac handles plausibly raise. Programming bugs
                # (NameError, ImportError) propagate.
                logger.debug("queued action failed", exc_info=True)
        # 2. Worker actions include the recording capture (its own
        # converge + grab), so only render here when the sim is IDLE:
        # keeps the live preview fresh between episodes without competing
        # with the recorder mid-episode.
        if n_actions == 0 and render:
            self._converge_render(self._idle_converge)
        # 3. Refresh joint-state cache for every robot.
        for rname, r in self._robots.items():
            if r.articulation is None:
                continue
            try:
                q = r.articulation.get_joint_positions()
                if q is not None:
                    arr = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
                    self._joint_cache[rname] = {jn: float(v) for jn, v in zip(r.joint_names, list(arr))}
            except (RuntimeError, ValueError, AttributeError, TypeError):
                pass
        # 4. Refresh camera frame cache only when we rendered this tick
        # (idle path); after actions, the capture already published its
        # frames to the cache.
        if render and n_actions == 0 and self._pump_cameras:
            for cname, cam in self._cameras.items():
                if cam.handle is None:
                    continue
                try:
                    img = self._grab_frame(cname, cam.handle)
                    if img is not None:
                        self._frame_cache[cname] = img
                except (RuntimeError, ValueError, AttributeError, TypeError):
                    logger.debug("pump frame grab failed for %s", cname, exc_info=True)

    def run_pump_forever(self, stop_event: Any = None) -> None:
        """Block on the MAIN THREAD running ``pump()`` in a loop.

        Drains queued worker actions (an executing episode) every
        iteration so the episode runs at full speed, and refreshes the
        live preview only every ``_idle_render_period`` IDLE seconds.
        A short sleep when idle keeps the renderer from running flat
        out -- which otherwise starves the Gradio HTTP thread so the
        page never loads.

        ``stop_event`` is a ``threading.Event``-style object whose
        ``is_set()`` returning truthy ends the loop. ``None`` (default)
        loops until ``KeyboardInterrupt``.
        """
        last_idle_render = 0.0
        self._pump_running = True
        try:
            while stop_event is None or not stop_event.is_set():
                # A whole-job submission (UI record/plan) takes priority:
                # run it inline on this main thread. The job drives the
                # sim directly (no per-frame round-trips); the preview
                # just freezes for its duration, which is the right
                # trade for a fast, reliable record.
                try:
                    job = self._main_jobs.get_nowait()
                except queue.Empty:
                    job = None
                if job is not None:
                    job()
                    last_idle_render = 0.0
                    continue
                busy = not self._action_q.empty()
                if busy:
                    self.pump(render=False)
                    continue
                now = time.time()
                do_render = (now - last_idle_render) >= self._idle_render_period
                self.pump(render=do_render)
                if do_render:
                    last_idle_render = now
                time.sleep(0.05)
        finally:
            self._pump_running = False

    def run_on_main(self, fn: Any, timeout: float | None = None) -> Any:
        """Run ``fn()`` on the MAIN THREAD (the pump owner) and return its result.

        Submitting a WHOLE job (rather than per-frame calls round-tripping
        through the action queue) lets the pump run it inline on the main
        thread, where ``fn`` drives the sim directly. While the job runs,
        the pump's normal loop is paused. Re-raises any exception from
        ``fn`` on the caller's thread; raises ``TimeoutError`` if the pump
        doesn't finish within ``timeout``. If already on the main thread,
        runs ``fn`` immediately.
        """
        if self._on_main_thread():
            return fn()
        done = threading.Event()
        box: dict[str, Any] = {}

        def _job() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
                box["exc"] = exc
            finally:
                done.set()

        self._main_jobs.put(_job)
        if not done.wait(timeout=timeout):
            raise TimeoutError("run_on_main timed out waiting for the main-thread pump.")
        if "exc" in box:
            raise box["exc"]
        return box.get("result")

    # --- joint targets / kinematic teleport --------------------------------

    def set_joint_positions(
        self,
        positions: Any = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Drive an articulated robot kinematically to ``positions``.

        Writes joint state directly (teleport), unlike ``send_action``'s
        PD position targets -- needed to replay trajectories on an
        actuator-less arm. ``positions`` may be a ``dict`` keyed by joint
        name (only the listed joints are written) or a list/array in the
        robot's joint order.
        """
        with self._lock:
            if not self._world_created or not self._robots:
                return {"status": "error", "content": [{"text": "No world/robot."}]}
            if positions is None:
                return {"status": "error", "content": [{"text": "'positions' is required."}]}
            if robot_name is None:
                robot_name = next(iter(self._robots))
            r = self._robots.get(robot_name)
            if r is None or r.articulation is None:
                return {"status": "error", "content": [{"text": f"Robot {robot_name!r} not initialized."}]}

            def _apply() -> None:
                if isinstance(positions, dict):
                    cur = list(r.articulation.get_joint_positions())
                    idx = {jn: i for i, jn in enumerate(r.joint_names)}
                    for jn, v in positions.items():
                        if jn in idx:
                            cur[idx[jn]] = float(v)
                    r.articulation.set_joint_positions(np.array(cur, dtype=float))
                else:
                    r.articulation.set_joint_positions(np.array(positions, dtype=float))

            if self._on_main_thread():
                _apply()
                return {"status": "success", "content": [{"text": "Set joint positions (main)."}]}
            self._action_q.put(_apply)
            return {"status": "success", "content": [{"text": "Set joint positions (queued)."}]}

    def move_object(
        self,
        name: str,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Teleport an object to ``(position, orientation)``.

        Velocities are zeroed so a teleport doesn't fling a dynamic body.
        """
        obj = self._objects.get(name)
        if obj is None or obj.handle is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        try:
            pos = np.array(position[:3], dtype=float) if position else None
            ori = np.array(orientation[:4], dtype=float) if orientation else None
            obj.handle.set_world_pose(position=pos, orientation=ori)
            if hasattr(obj.handle, "set_linear_velocity"):
                obj.handle.set_linear_velocity(np.zeros(3))
            if hasattr(obj.handle, "set_angular_velocity"):
                obj.handle.set_angular_velocity(np.zeros(3))
            # Also write the USD xform translate/orient DIRECTLY:
            # ``set_world_pose`` updates the PhysX/fabric transform, but
            # the RENDER reads the prim's USD ``xformOp:translate``, and
            # the fabric->USD writeback can lag for a teleported body so
            # it renders at a stale pose. Writing the xform ops keeps the
            # rendered mesh exactly at the commanded pose.
            if pos is not None:
                try:
                    import omni.usd  # type: ignore[import-not-found]
                    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(obj.prim_path)
                    if prim and prim.IsValid():
                        xf = UsdGeom.Xformable(prim)
                        top = None
                        for op in xf.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                top = op
                                break
                        if top is None:
                            top = xf.AddTranslateOp()
                        top.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                        if ori is not None:
                            for op in xf.GetOrderedXformOps():
                                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                    op.Set(Gf.Quatf(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])))
                                    break
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    logger.debug("move_object USD xform write skipped", exc_info=True)
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            return {"status": "error", "content": [{"text": f"move_object failed: {type(exc).__name__}: {exc}"}]}
        return {"status": "success", "content": [{"text": f"'{name}' moved to {position or 'same'}."}]}

    def set_object_kinematic(self, name: str, kinematic: bool = True) -> dict[str, Any]:
        """Toggle an object's rigid body between KINEMATIC and dynamic.

        A dynamic body's transform is owned by the physics solver (even
        with its collider disabled), so a teleported pose can render
        drifted; a KINEMATIC body takes its transform straight from
        ``set_world_pose`` and renders faithfully. Flip a carried object
        kinematic, restore to dynamic on release. Toggles
        ``UsdPhysics.RigidBodyAPI.kinematicEnabled``; best-effort.
        """
        obj = self._objects.get(name)
        if obj is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        # Prefer the wrapper's own setter if present.
        if obj.handle is not None:
            for meth in ("set_rigid_body_kinematic", "set_kinematic_enabled"):
                fn = getattr(obj.handle, meth, None)
                if callable(fn):
                    try:
                        fn(bool(kinematic))
                        return {"status": "success", "content": [{"text": f"'{name}' kinematic={kinematic}."}]}
                    except (RuntimeError, ValueError, AttributeError, TypeError):
                        logger.debug("%s failed; trying USD API", meth, exc_info=True)
        # Fallback: toggle UsdPhysics.RigidBodyAPI kinematicEnabled directly.
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdPhysics  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.prim_path)
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath()) or UsdPhysics.RigidBodyAPI.Apply(prim)
            attr = api.GetKinematicEnabledAttr()
            if not attr:
                attr = api.CreateKinematicEnabledAttr()
            attr.Set(bool(kinematic))
            return {
                "status": "success",
                "content": [{"text": f"'{name}' kinematic={kinematic} (USD)."}],
            }
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError) as exc:
            return {
                "status": "error",
                "content": [{"text": f"set_object_kinematic failed: {type(exc).__name__}: {exc}"}],
            }

    def set_object_collision(self, name: str, enabled: bool = True) -> dict[str, Any]:
        """Enable / disable an object's collider (keeps the visual mesh intact).

        Used by the kinematic teleport-grasp: a carried object's collider
        would interpenetrate the closing gripper fingers and the contact
        forces fling the stiff PD arm into oscillation. Disable while
        grasped, re-enable on release.
        """
        obj = self._objects.get(name)
        if obj is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        if obj.handle is not None:
            try:
                obj.handle.set_collision_enabled(bool(enabled))
                return {"status": "success", "content": [{"text": f"'{name}' collision {'on' if enabled else 'off'}."}]}
            except (RuntimeError, ValueError, AttributeError, TypeError):
                logger.debug("set_collision_enabled unavailable; falling back to USD API", exc_info=True)
        # Fallback: toggle UsdPhysics.CollisionAPI on the prim directly.
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdPhysics  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.prim_path)
            api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()) or UsdPhysics.CollisionAPI.Apply(prim)
            api.GetCollisionEnabledAttr().Set(bool(enabled))
            return {
                "status": "success",
                "content": [{"text": f"'{name}' collision {'on' if enabled else 'off'} (USD)."}],
            }
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError) as exc:
            return {
                "status": "error",
                "content": [{"text": f"set_object_collision failed: {type(exc).__name__}: {exc}"}],
            }

    def _object_position(self, name: str) -> list[float] | None:
        """Return the world-frame position of ``name`` (or ``None`` if missing)."""
        obj = self._objects.get(name)
        if obj is None or obj.handle is None:
            return None
        try:
            pos, _ = obj.handle.get_world_pose()
            return [float(x) for x in pos]
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return None

    def gripper_frame_pos(self, robot_name: str | None = None) -> list[float] | None:
        """World position of the robot's gripper / tool link (translation only)."""
        pose = self.gripper_frame_pose(robot_name)
        return pose[0] if pose else None

    def gripper_frame_pose(self, robot_name: str | None = None) -> tuple[list[float], list[float]] | None:
        """World pose of the robot's gripper / tool link: ``(translation, rotation)``.

        ``translation`` is the link origin in world coords; ``rotation``
        is the row-major 3x3 (flattened to 9) whose *columns* are the
        tool frame's local x/y/z axes in world coords, so
        ``world = R @ local``. Used to seat a carried object rigidly in
        the tool frame. Prefers a ``gripper_frame``/``tool`` link, then
        any ``gripper``/``moving_jaw`` link, under the robot's prim
        subtree.
        """
        if robot_name is None:
            robot_name = next(iter(self._robots), None)
        r = self._robots.get(robot_name) if robot_name else None
        if r is None:
            return None
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import (  # type: ignore[import-not-found]
                Gf,
                Sdf,
                Usd,
                UsdGeom,
            )

            stage = omni.usd.get_context().get_stage()
            # ``r.actual_prim_path`` is where the importer actually placed
            # the robot (may differ from the requested ``prim_path``).
            # Walk up to the top-level robot prim and search its subtree
            # for the gripper / tool link.
            sdf_path = Sdf.Path(r.actual_prim_path)
            top = sdf_path
            while top.GetParentPath() != Sdf.Path.absoluteRootPath and top.GetParentPath() != Sdf.Path.emptyPath:
                top = top.GetParentPath()
            root = stage.GetPrimAtPath(top)
            if not root or not root.IsValid():
                return None
            preferred = None
            fallback = None
            for p in Usd.PrimRange(root):
                if not p.IsA(UsdGeom.Xformable):
                    continue
                ln = p.GetName().lower()
                if "gripper_frame" in ln or "tool" in ln:
                    preferred = p
                    break
                if "moving_jaw" in ln or "gripper" in ln:
                    fallback = fallback or p
            prim = preferred or fallback
            if prim is None:
                return None
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = xf.ExtractTranslation()

            def _axis(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
                d = xf.TransformDir(Gf.Vec3d(vx, vy, vz))
                n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5 or 1.0
                return (d[0] / n, d[1] / n, d[2] / n)

            ax = _axis(1.0, 0.0, 0.0)
            ay = _axis(0.0, 1.0, 0.0)
            az = _axis(0.0, 0.0, 1.0)
            rot = [ax[0], ay[0], az[0], ax[1], ay[1], az[1], ax[2], ay[2], az[2]]
            pos = [float(t[0]), float(t[1]), float(t[2])]
            return pos, [float(x) for x in rot]
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError):
            logger.debug("gripper_frame_pose failed", exc_info=True)
            return None

    # --- DLSS-ghost mitigation + RTX renderer config ------------------------

    def _refresh_all_render_products(self, n: int = 1) -> None:
        """Tick the renderer ``n`` times so EVERY camera's RTX render product
        accumulates a fresh frame.

        A single ``world.step(render=True)`` only reliably refreshes the
        PRIMARY render product; an extra render-only tick lets Kit flush
        the secondary products before read-back. Kept deliberately LIGHT:
        ``SimulationApp.update()`` renders without advancing physics
        (falling back to one ``world.step(render=True)``); heavier
        per-frame render loops overloaded the RTX Hydra-texture pipeline
        and crashed Kit over long sessions. Main-thread only (renderer
        constraint).
        """
        if not self._world_created or self._world is None:
            return
        # Prefer SimulationApp.update(); fall back to world.step(render=True).
        app = getattr(self, "_app", None)
        update = getattr(app, "update", None) if app is not None else None
        for _ in range(max(1, n)):
            if callable(update):
                update()
            else:
                self._world.step(render=True)

    def _converge_render(self, n: int = 8) -> None:
        """Render ``n`` ticks while HOLDING the robots at their current pose.

        Re-asserts each robot's joint positions (and zeroes velocities)
        before every render so the pose stays frozen and the DLSS temporal
        upscaler converges on a single static image instead of ghosting a
        drifting arm.
        """
        if not self._world_created or self._world is None:
            return
        for _ in range(max(1, n)):
            for r in self._robots.values():
                if r.articulation is None:
                    continue
                try:
                    q = r.articulation.get_joint_positions()
                    if q is not None:
                        qa = np.asarray(q, dtype=float)
                        r.articulation.set_joint_positions(qa)
                        try:
                            r.articulation.set_joint_velocities(np.zeros_like(qa))
                        except (RuntimeError, ValueError, AttributeError, TypeError):
                            pass
                except (RuntimeError, ValueError, AttributeError, TypeError):
                    pass
            self._world.step(render=True)

    def _grab_frame(self, cname: str, cam: Any) -> Any:
        """Capture ``cam`` as an RGB uint8 array at the camera's requested output size.

        The RTX camera renders at a higher native resolution (to keep
        DLSS out of its temporal-ghost regime); this downscales the
        result back to the size the caller asked for. Returns ``None``
        if no frame is available yet.
        """
        frame = cam.get_rgba()
        if frame is None or not getattr(frame, "size", 0):
            return None
        img = np.asarray(frame)[:, :, :3].astype("uint8")
        out = self._cam_out_size.get(cname)
        if out is not None:
            ow, oh = out
            if img.shape[1] != ow or img.shape[0] != oh:
                img = self._resize_rgb(img, ow, oh)
        return img

    @staticmethod
    def _resize_rgb(img: Any, out_w: int, out_h: int) -> Any:
        """Downscale an HxWx3 uint8 array to ``(out_h, out_w)``.

        Uses cv2 / PIL if present, else a fast NumPy area-average /
        nearest fallback (no new deps).
        """
        try:
            import cv2  # type: ignore[import-not-found]

            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            pass
        try:
            from PIL import Image  # type: ignore[import-not-found]

            resample = getattr(Image, "Resampling", Image).BILINEAR
            return np.asarray(Image.fromarray(img).resize((out_w, out_h), resample))
        except ImportError:
            pass
        h, w = img.shape[:2]
        if w % out_w == 0 and h % out_h == 0:
            fx, fy = w // out_w, h // out_h
            return img.reshape(out_h, fy, out_w, fx, 3).mean(axis=(1, 3)).astype("uint8")
        ys = (np.arange(out_h) * (h / out_h)).astype(int).clip(0, h - 1)
        xs = (np.arange(out_w) * (w / out_w)).astype(int).clip(0, w - 1)
        return img[ys][:, xs]

    def _configure_renderer(self) -> None:
        """Best-effort RTX settings for a stable real-time image.

        These carb settings nudge RTX toward a single-frame-stable image,
        but the pipeline re-asserts ``/rtx/post/aa/op`` back to DLSS on
        every render tick, so they do NOT by themselves stop the
        moving-arm "ghost" -- the actual fix is the high native render
        resolution (>= ``_MIN_RENDER_PX`` wide) plus ``_converge_render``.
        Skipped silently when ``carb.settings`` isn't importable.
        """
        try:
            import carb  # type: ignore[import-not-found]

            s = carb.settings.get_settings()
            s.set("/rtx/rendermode", "RaytracedLighting")
            s.set("/rtx/directLighting/sampledLighting/enabled", True)
            s.set("/rtx/raytracing/subframes", 1)
            s.set("/rtx/pathtracing/totalSpp", 1)
            s.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
            s.set("/rtx/post/aa/op", 1)
            s.set("/rtx/post/dlss/execMode", 0)
            s.set("/rtx/post/taa/enabled", False)
            s.set("/rtx/directLighting/denoiser/enabled", False)
            s.set("/rtx/raytracing/lightcache/spatialCache/enabled", False)
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("renderer config skipped", exc_info=True)

    def _add_lighting(self) -> None:
        """Add a dome + key + fill light so RTX camera frames aren't black.

        Unlike MuJoCo (which has implicit headlight / ambient), an Isaac
        stage is unlit by default -- without this, ``get_rgba()``
        returns near-black frames and the UI preview looks empty.
        Best-effort; skipped silently when Pixar USD imports fail.
        """
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import (  # type: ignore[import-not-found]
                Gf,
                Sdf,
                UsdGeom,
                UsdLux,
            )

            stage = omni.usd.get_context().get_stage()
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome"))
            dome.CreateIntensityAttr(800.0)
            distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/key"))
            distant.CreateIntensityAttr(2500.0)
            distant.CreateAngleAttr(1.0)
            UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))
            fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/fill"))
            fill.CreateIntensityAttr(1500.0)
            fill.CreateAngleAttr(1.0)
            UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 0.0, 180.0))
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Could not add scene lighting", exc_info=True)

    def describe(self) -> dict[str, Any]:
        """Return the Isaac engine's live discovery surface.

        Extends the base :meth:`SimEngine.describe` contract with the backend
        identity, the registered RTX camera names, world state, and the
        LeRobotDataset recording family
        (:class:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin`)
        so an agent enumerating ``describe()["methods"]`` discovers the
        record-and-stream workflow (``start_recording`` -> ``run_policy`` ->
        ``save_episode`` -> ``stop_recording`` -> ``stream_dataset``) exactly
        as it does on the MuJoCo and Newton backends.
        """
        desc = super().describe()
        desc["backend"] = "isaac"
        desc["cameras"] = sorted(self._cameras)
        desc["world_created"] = self._world_created
        desc["methods"].update(
            {
                "add_camera": (
                    "(name='default', position=None, target=None, width=None, "
                    "height=None, fov=60.0) -> dict  # register an RTX camera "
                    "(rendered frames ride get_observation and recordings)"
                ),
                "remove_camera": "(name: str) -> dict  # remove a registered RTX camera",
                "start_recording": (
                    "(repo_id='local/sim_recording', task='', fps=30, root=None, "
                    "push_to_hub=False, vcodec='h264', overwrite=False, cameras=None) -> dict  "
                    "# record joint state + action + RTX cameras to a LeRobotDataset "
                    "(needs render_mode='rtx_realtime' for camera columns)"
                ),
                "save_episode": (
                    "() -> dict  # flush the current rollout as one episode; prefer "
                    "run_policy(n_episodes=N) which flushes a boundary per episode"
                ),
                "stop_recording": "(push_to_hub=False, bucket=None, run_id=None) -> dict",
                "get_recording_status": "() -> dict",
                "stream_dataset": (
                    "(repo_id: str, **kwargs) -> StreamingDatasetReader  # lazily read a "
                    "recorded LeRobotDataset back (root=, episodes=, delta_timestamps=, ...)"
                ),
                "verify_dataset_episodes": (
                    "(expected: int) -> dict  # after stop_recording, read the parquet and "
                    "confirm the dataset holds exactly `expected` episodes; status=error on mismatch"
                ),
                "start_cameras_recording": (
                    "(cameras=None, output_dir=None, fps=30, name=None, max_frames_per_camera=3000) -> dict  "
                    "# raw per-camera MP4 capture (no lerobot dependency)"
                ),
                "stop_cameras_recording": "() -> dict  # finalize the raw MP4 capture",
            }
        )
        return desc

    def cleanup(self) -> None:
        """Release all resources.

        Callers must invoke this explicitly (or use the class as a context
        manager). There is intentionally no ``__del__`` finalizer: at
        interpreter shutdown the ``threading`` / ``logger`` / ``omni``
        modules can already be partially torn down, and acquiring
        ``self._lock`` from a finalizer is unsafe. Relying on GC for
        Isaac Sim cleanup also leaks the ``World``/USD stage on the
        common case where the GC scheduler defers the finalizer past
        the SimulationApp shutdown.
        """
        if self._world_created:
            self.destroy()

    def __enter__(self) -> IsaacSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"IsaacSimulation("
            f"num_envs={self._config.num_envs}, "
            f"device={self._config.device!r}, "
            f"headless={self._config.headless}, "
            f"world={'created' if self._world_created else 'none'})"
        )
