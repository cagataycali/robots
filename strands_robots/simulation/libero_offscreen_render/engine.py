"""LIBERO OffScreenRenderEnv-backed simulation engine.

Wraps NVIDIA's upstream ``libero.libero.envs.OffScreenRenderEnv`` (the
robosuite-derived MuJoCo env that LIBERO datasets were generated against)
behind the strands-robots :class:`SimEngine` interface, so
``evaluate_benchmark`` works against the upstream LIBERO setup without
the need for our auto-generated scene MJCF + custom OSC controller path.

This engine exists because rounds 36-41 verified-correct fixes
(action_horizon, max_steps, seeding, image V-flip, image dims, gripper
RLDS→robosuite polarity) didn't move ``success_rate`` off 0 against
``nvidia/GR00T-N1.7-LIBERO``, while NVIDIA's own reference eval
(``run_gr00t_sim_policy``) gets ``success_rate = 1.0`` in 54s for 5 eps
on the same checkpoint+task. Round-42 step-by-step instrumentation
(``/tmp/opencode/eval-runs/instrument_*.py``) measured a 50× action
divergence at near-identical state inputs, suggesting the residual gap
is in our scene+controller path that the structural fixes couldn't
close.

Round 43 sidesteps the gap by routing through the upstream env directly.

NOT FOR GENERAL USE: this engine ONLY supports LIBERO tasks. It rejects
all the SimEngine methods that don't make sense for the upstream env
(``add_object``, ``randomize``, etc.) by raising ``NotImplementedError``.
For general MuJoCo simulation use ``MuJoCoSimEngine``.

Lifecycle:

1. ``create_world()`` — no-op (env created lazily on first task setup).
2. ``setup_libero_task(bddl_file_name, init_state=None)`` —
   constructs the underlying ``OffScreenRenderEnv`` from BDDL.
3. ``reset()`` — resets the env (+ optionally applies ``init_state``).
4. ``send_action(action_dict, robot_name)`` — packs action_dict into
   the 7-vector ``[x, y, z, roll, pitch, yaw, gripper]``, applies
   NVIDIA's ``normalize_gripper_action + invert_gripper_action`` chain
   (matches upstream ``LiberoEnv.step``), then calls ``env.step``.
5. ``get_observation()`` — pulls ``robot0_eef_pos/quat/gripper_qpos`` +
   ``agentview_image`` + ``robot0_eye_in_hand_image`` from the raw
   robosuite obs and exposes them under our flat schema (``x``, ``y``,
   ``z``, ``roll``, ``pitch``, ``yaw``, ``gripper``, ``image``,
   ``wrist_image``). Image V-flip and gripper conventions match
   upstream's ``_process_observation`` output, so policies trained on
   upstream LIBERO data see exactly the inputs they were trained on.
6. ``destroy()`` — closes the underlying env.

Round 43 (#168). See PR #168 round-43 commit message for the full
bisect history that motivated this engine.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


def _normalize_gripper_action(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """[0, 1] → [-1, +1] then binarize via ``np.sign``.

    Mirrors NVIDIA's upstream
    ``Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py:normalize_gripper_action``.
    """
    out = np.array(action, dtype=np.float64, copy=True)
    out[..., -1] = 2 * out[..., -1] - 1
    if binarize:
        out[..., -1] = np.sign(out[..., -1])
    return out


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper action sign (RLDS → LIBERO/robosuite convention).

    Mirrors NVIDIA's upstream
    ``Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py:invert_gripper_action``.
    """
    out = np.array(action, dtype=np.float64, copy=True)
    out[..., -1] *= -1.0
    return out


def _quat_xyzw_to_rpy_xyz(quat_xyzw: np.ndarray) -> tuple[float, float, float]:
    """Convert (x, y, z, w) quaternion → extrinsic XYZ Euler (roll, pitch, yaw).

    Same axes='sxyz' convention upstream LIBERO uses. Round-32 of
    ``LiberoAdapter`` does the same conversion from a (w, x, y, z) quat
    via ``_quat_wxyz_to_rpy_xyz``; we reorder here because robosuite's
    ``robot0_eef_quat`` is xyzw.
    """
    x, y, z, w = (float(q) for q in quat_xyzw)
    # Standard atan2 + asin Euler decomposition (extrinsic XYZ).
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


class LiberoOffScreenRenderEngine(SimEngine):
    """LIBERO-only :class:`SimEngine` backend that delegates to upstream
    ``OffScreenRenderEnv``.

    See module docstring for design rationale and lifecycle.
    """

    def __init__(
        self,
        tool_name: str = "libero_offscreen_render",
        camera_height: int = 256,
        camera_width: int = 256,
        **kwargs: Any,
    ):
        """Construct the engine. No env is created until
        :meth:`setup_libero_task` is called.

        Args:
            tool_name: Identifier surfaced to the agent layer; mirrors
                ``MuJoCoSimEngine.tool_name`` for plugin-interface
                compatibility.
            camera_height: Render height for ``agentview`` and
                ``robot0_eye_in_hand``. Must match training distribution
                — defaults to 256 (NVIDIA's training resolution).
            camera_width: Render width. Same constraint.
            **kwargs: Ignored (forward-compatibility with the
                ``MuJoCoSimEngine`` constructor signature).
        """
        super().__init__()
        self.tool_name_str = tool_name
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)

        # Lazy-loaded upstream env. Set by :meth:`setup_libero_task`.
        self._env: Any = None
        self._task_bddl_path: str | None = None
        self._init_state: np.ndarray | None = None
        # Cache of the latest observation from env.reset() / env.step().
        # ``OffScreenRenderEnv`` doesn't expose ``observation_spec()``;
        # the obs is only returned from step/reset. We cache it here so
        # :meth:`get_observation` (which the eval loop calls between
        # ``send_action`` calls) can return the latest snapshot without
        # re-stepping.
        self._latest_obs: dict[str, Any] = {}

        # Strands-robots-style world snapshot (mostly a placeholder so
        # plugin interface methods can return something).
        self._world: SimWorld | None = None

        # Mark we have a robot (the Panda is implicit in OffScreenRenderEnv).
        # Populated on setup_libero_task so list_robots() and
        # robot_joint_names() can return.
        self._robot_joints: list[str] = []

    # World lifecycle

    def create_world(
        self,
        timestep: float | None = None,  # noqa: ARG002 - upstream env owns timestep
        gravity: list[float] | None = None,  # noqa: ARG002
        ground_plane: bool = True,  # noqa: ARG002
    ) -> dict[str, Any]:
        """No-op. The upstream ``OffScreenRenderEnv`` is created lazily
        on :meth:`setup_libero_task` (not here) because it needs a BDDL
        path that this engine can't know without a benchmark spec.
        """
        # Initialize a placeholder SimWorld so list_robots() doesn't
        # crash before setup_libero_task. Simulation.create_world
        # contract is "world ready for add_robot".
        self._world = SimWorld()
        return {
            "status": "success",
            "content": [{"text": "🌍 LIBERO world created (env constructed lazily on benchmark setup)."}],
        }

    def destroy(self) -> dict[str, Any]:
        """Tear down the underlying env if any, release world."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning("LiberoOffScreenRenderEngine.destroy: env.close raised %s", e)
            self._env = None
        self._world = None
        return {"status": "success", "content": [{"text": "🗑️ LIBERO env destroyed."}]}

    def reset(self) -> dict[str, Any]:
        """Reset the underlying env. Re-applies ``init_state`` if one
        was provided to :meth:`setup_libero_task`.
        """
        if self._env is None:
            return {
                "status": "error",
                "content": [{"text": "reset: env not initialized. Call setup_libero_task first."}],
            }
        try:
            obs = self._env.reset()
            if self._init_state is not None:
                self._env.set_init_state(self._init_state)
                # set_init_state writes qpos/qvel directly without
                # producing a new obs; do one zero-action step so
                # observables update with the init state. NVIDIA's
                # __main__ debug block does the same (libero_env.py:257).
                obs, _, _, _ = self._env.step(np.zeros(7))
            self._latest_obs = obs if isinstance(obs, dict) else {}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "content": [{"text": f"reset: {e}"}]}
        return {"status": "success", "content": [{"text": "🔄 LIBERO env reset."}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance physics by ``n_steps`` zero-action steps.

        Implements the SimEngine.step contract. Used by callers that
        want to settle physics without sending an action; here we
        synthesize a 7-vector of zeros (which after the gripper
        conversion becomes "close intent" — the same as upstream's
        ``LiberoEnv.step({...zeros...})`` would produce). Most LIBERO
        eval flows don't call this directly; ``send_action`` is the
        primary advance path.
        """
        if self._env is None:
            return {"status": "error", "content": [{"text": "step: env not initialized."}]}
        try:
            for _ in range(int(n_steps)):
                obs, _, _, _ = self._env.step(np.zeros(7))
                if isinstance(obs, dict):
                    self._latest_obs = obs
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "content": [{"text": f"step: {e}"}]}
        return {"status": "success", "content": [{"text": f"⏩ stepped {n_steps}× (zero-action)."}]}

    def get_state(self) -> dict[str, Any]:
        """Minimal state summary."""
        if self._env is None:
            return {"status": "error", "content": [{"text": "get_state: env not initialized."}]}
        return {
            "status": "success",
            "content": [
                {"text": f"LIBERO env loaded: {self._task_bddl_path}"},
                {"json": {"task_bddl": self._task_bddl_path, "robots": self.list_robots()}},
            ],
        }

    # Robot management

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,  # noqa: ARG002
        data_config: str | None = None,  # noqa: ARG002
        position: list[float] | None = None,  # noqa: ARG002
        orientation: list[float] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """No-op. The Panda is implicit in the LIBERO scene MJCF.

        Returns success so :meth:`evaluate_benchmark`'s pre-flight
        ("No robots in sim") check passes.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "add_robot: call create_world first."}]}
        # Track the robot name so list_robots() reports it; the actual
        # joints are populated on setup_libero_task.
        self._world.robots[name] = SimRobot(
            name=name,
            urdf_path="<libero-offscreen-render>",  # synthetic — the env owns the actual MJCF
            joint_names=list(self._robot_joints),
        )
        return {
            "status": "success",
            "content": [{"text": f"🤖 Robot {name!r} registered (Panda is implicit in LIBERO scene)."}],
        }

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove the robot registration (env-side robot stays — it's
        defined by the BDDL/MJCF)."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "remove_robot: world not created."}]}
        self._world.robots.pop(name, None)
        return {"status": "success", "content": [{"text": f"🗑️ Robot {name!r} unregistered."}]}

    def list_robots(self) -> list[str]:
        """Return registered robot names."""
        if self._world is None:
            return []
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return joint names for ``robot_name``.

        Pulled from the underlying env's MuJoCo model after
        :meth:`setup_libero_task`. Before that, returns an empty list.
        """
        if self._world is None:
            return []
        robot = self._world.robots.get(robot_name)
        if robot is None:
            return []
        return list(robot.joint_names)

    # Object management — not supported (LIBERO objects come from BDDL).

    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "error",
            "content": [{"text": "add_object: LIBERO scene objects come from BDDL, not runtime add_object. No-op."}],
        }

    def remove_object(self, name: str) -> dict[str, Any]:  # noqa: ARG002
        return {
            "status": "error",
            "content": [{"text": "remove_object: LIBERO scene objects come from BDDL. No-op."}],
        }

    # Observation / Action

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:  # noqa: ARG002
        """Translate robosuite obs → strands-robots flat schema.

        Returns:
            Dict with keys matching what :class:`LiberoAdapter`'s
            ``augment_observation`` produces in the MuJoCo backend
            path: bare keys ``x`` / ``y`` / ``z`` / ``roll`` /
            ``pitch`` / ``yaw`` / ``gripper`` (state) and ``image`` /
            ``wrist_image`` (uint8 H×W×3, V-flipped to match upstream
            convention). Plus joint readings under their bare names.
        """
        if self._env is None:
            return {}

        # ``OffScreenRenderEnv`` doesn't expose ``observation_spec()``;
        # the obs is only returned from step/reset. Use the cache.
        raw = self._latest_obs
        if not isinstance(raw, dict) or not raw:
            logger.debug(
                "LiberoOffScreenRenderEngine.get_observation: no cached obs "
                "(env reset/step never produced one). Returning empty dict."
            )
            return {}

        obs: dict[str, Any] = {}

        # State: EEF pose + gripper qpos (the seven keys the
        # libero_panda data_config expects).
        eef_pos = raw.get("robot0_eef_pos")
        if eef_pos is not None:
            obs["x"] = float(eef_pos[0])
            obs["y"] = float(eef_pos[1])
            obs["z"] = float(eef_pos[2])
        eef_quat = raw.get("robot0_eef_quat")
        if eef_quat is not None:
            roll, pitch, yaw = _quat_xyzw_to_rpy_xyz(eef_quat)
            obs["roll"] = roll
            obs["pitch"] = pitch
            obs["yaw"] = yaw
        gripper_qpos = raw.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            obs["gripper"] = [float(v) for v in gripper_qpos]

        # Joint state — populate under bare joint names so the policy's
        # ``set_robot_state_keys`` mapping works.
        joint_qpos = raw.get("robot0_joint_pos")
        if joint_qpos is not None:
            for i, name in enumerate(self._robot_joints):
                if i < len(joint_qpos):
                    obs[name] = float(joint_qpos[i])

        # Images: V-flipped agentview + robot0_eye_in_hand. Upstream's
        # ``OffScreenRenderEnv`` returns OpenGL bottom-row-zero
        # convention; we just reverse the H axis here so producer output
        # matches OpenGL framebuffer convention. The policy's
        # ``image_rotation_180`` flag (set on ``libero_panda``'s
        # ``Gr00tDataConfig``) then applies the second flip to convert
        # OpenGL → training convention. This matches the contract the
        # ``LiberoAdapter`` (MuJoCo path) uses since round 39.
        #
        # Pre-#169 this baked the full ``[::-1, ::-1]`` rotation here,
        # producing training-convention output directly. That worked for
        # LOCAL inference (which doesn't apply the policy-side rotation
        # flag) but DOUBLE-rotated in SERVICE mode (engine 180° + policy
        # 180° = identity = OpenGL OOD), causing #169's
        # ``success_rate=0`` against the docker GR00T server. The fix
        # moves the second flip into the policy's
        # ``_build_service_observation`` /
        # ``_prepare_observation`` (now consistent across
        # transports) so this engine just produces OpenGL convention
        # like every other observation producer.
        if not skip_images:
            agent_img = raw.get("agentview_image")
            if isinstance(agent_img, np.ndarray):
                obs["image"] = np.ascontiguousarray(agent_img[::-1, :])
            wrist_img = raw.get("robot0_eye_in_hand_image")
            if isinstance(wrist_img, np.ndarray):
                obs["wrist_image"] = np.ascontiguousarray(wrist_img[::-1, :])

        return obs

    def send_action(self, action: dict[str, Any], robot_name: str | None = None, n_substeps: int = 1) -> None:  # noqa: ARG002
        """Pack ``action`` dict → 7-vector → upstream ``env.step``.

        Mirrors NVIDIA's upstream ``LiberoEnv.step`` exactly:

        1. Concatenate 7 channels into a numpy array.
        2. ``normalize_gripper_action`` ([0, 1] → ±1).
        3. ``invert_gripper_action`` (RLDS → LIBERO/robosuite sign).
        4. ``self._env.step(action_vector)``.

        ``n_substeps`` is ignored — robosuite's controller has its own
        sub-stepping (25 substeps × 500 Hz = 20 Hz control), already
        matched to training.
        """
        if self._env is None:
            raise RuntimeError("send_action: env not initialized. Call setup_libero_task first.")

        def _to_scalar(v: Any) -> float:
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v).flatten()
                return float(arr[0]) if arr.size else 0.0
            return float(v)

        action_vector = np.array(
            [
                _to_scalar(action.get("x", 0.0)),
                _to_scalar(action.get("y", 0.0)),
                _to_scalar(action.get("z", 0.0)),
                _to_scalar(action.get("roll", 0.0)),
                _to_scalar(action.get("pitch", 0.0)),
                _to_scalar(action.get("yaw", 0.0)),
                _to_scalar(action.get("gripper", 0.0)),
            ],
            dtype=np.float64,
        )
        action_vector = _normalize_gripper_action(action_vector)
        action_vector = _invert_gripper_action(action_vector)

        try:
            obs, _, _, _ = self._env.step(action_vector)
            if isinstance(obs, dict):
                self._latest_obs = obs
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoOffScreenRenderEngine.send_action: env.step raised %s", e)

    # Rendering

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Return the latest cached image from
        :meth:`get_observation` for ``camera_name``.

        Resolution is fixed by the env's camera config (256×256 by
        default); ``width``/``height`` overrides are ignored.

        Supported camera names: ``image`` (= agentview),
        ``wrist_image`` (= robot0_eye_in_hand), and the bare upstream
        names (``agentview``, ``robot0_eye_in_hand``).
        """
        if self._env is None:
            return {"status": "error", "content": [{"text": "render: env not initialized."}]}

        obs = self.get_observation(skip_images=False)
        # Map alias names to internal observation keys.
        alias = {
            "default": "image",
            "agentview": "image",
            "robot0_eye_in_hand": "wrist_image",
            "robot0_eye_in_hand_image": "wrist_image",
        }
        key = alias.get(camera_name, camera_name)
        img = obs.get(key)
        if img is None:
            return {
                "status": "error",
                "content": [{"text": f"render: camera {camera_name!r} not in obs (keys={sorted(obs.keys())})."}],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"📸 {img.shape[1]}x{img.shape[0]} from {camera_name!r} (upstream OffScreenRenderEnv)"},
                {"image": {"format": "rgb_uint8", "ndarray": img}},
                {"json": {"camera": key}},
            ],
        }

    # LIBERO-specific lifecycle

    def setup_libero_task(self, task_bddl_file: str, init_state: np.ndarray | None = None) -> dict[str, Any]:
        """Construct the underlying ``OffScreenRenderEnv`` for ``task_bddl_file``.

        Idempotent: if the env is already set up for the same BDDL,
        this is a no-op.

        Args:
            task_bddl_file: Path to the LIBERO BDDL describing the task.
                Forwarded to ``OffScreenRenderEnv(bddl_file_name=...)``.
            init_state: Optional ``ndarray[(1+nq+nv,)]`` canonical init
                state. When set, :meth:`reset` calls
                ``env.set_init_state(init_state)`` after the upstream
                reset. Pass ``None`` to use the BDDL-default state
                (matching NVIDIA's ``run_gr00t_sim_policy`` flow).

        Returns:
            Standard status dict.
        """
        require_optional("libero", pip_install="libero", extra="benchmark-libero", purpose="LIBERO eval engine")
        require_optional("robosuite", pip_install="robosuite", extra="benchmark-libero", purpose="LIBERO eval engine")

        from libero.libero.envs import OffScreenRenderEnv

        if self._env is not None and self._task_bddl_path == task_bddl_file:
            # Same task — keep the env, just (re-)apply init_state.
            self._init_state = np.asarray(init_state) if init_state is not None else None
            return {
                "status": "success",
                "content": [{"text": f"📂 LIBERO env already loaded for {task_bddl_file!r}; init_state updated."}],
            }

        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
            self._env = None

        try:
            self._env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=self.camera_height,
                camera_widths=self.camera_width,
            )
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "content": [{"text": f"setup_libero_task: OffScreenRenderEnv failed: {e}"}]}

        self._task_bddl_path = task_bddl_file
        self._init_state = np.asarray(init_state) if init_state is not None else None

        # Discover joint names from the underlying robosuite model so
        # robot_joint_names() reports them.
        self._robot_joints = self._discover_robot_joints()
        # Update any registered SimRobot's joint list to match.
        if self._world is not None:
            for r in self._world.robots.values():
                r.joint_names = list(self._robot_joints)

        return {
            "status": "success",
            "content": [{"text": f"📂 LIBERO env loaded for {task_bddl_file!r} (joints={len(self._robot_joints)})."}],
        }

    def _discover_robot_joints(self) -> list[str]:
        """Read robot0_joint{1..7} names from the underlying model.

        Returns the canonical 7 LIBERO Panda arm joints in order. Used
        by :meth:`robot_joint_names` so policies that call
        ``set_robot_state_keys`` get a stable schema.
        """
        if self._env is None:
            return []
        try:
            import mujoco  # type: ignore[import-not-found]
        except ImportError:
            return []
        sim = getattr(self._env, "sim", None)
        if sim is None:
            return []
        model = getattr(sim, "model", None) or getattr(self._env, "_model", None)
        if model is None:
            return []
        names: list[str] = []
        for i in range(1, 8):
            jname = f"robot0_joint{i}"
            try:
                jid = mujoco.mj_name2id(
                    model._model if hasattr(model, "_model") else model, mujoco.mjtObj.mjOBJ_JOINT, jname
                )
            except Exception:  # noqa: BLE001
                jid = -1
            if jid >= 0:
                names.append(jname)
        return names

    # SimEngine "optional override" methods that don't apply

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Reject — LIBERO scenes are loaded via :meth:`setup_libero_task`
        with a BDDL path, not a generic MJCF/scene path."""
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "load_scene: LiberoOffScreenRenderEngine doesn't support generic scene loading. "
                        "Use setup_libero_task(bddl_file_name=...) to load a LIBERO task."
                    )
                }
            ],
        }

    def randomize(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": "randomize: not supported on LiberoOffScreenRenderEngine."}]}

    def get_contacts(self) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": "get_contacts: not supported on LiberoOffScreenRenderEngine."}]}

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
            self._env = None
