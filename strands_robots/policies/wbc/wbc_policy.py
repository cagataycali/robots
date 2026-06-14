"""WBC Policy - ONNX inference for humanoid whole-body locomotion.

Implements the Policy ABC by running real ONNX inference from the
``nvidia/GR00T-WholeBodyControl`` HuggingFace checkpoint. The policy
maintains observation history (6-frame window of 86-dim single-step
observations) and auto-switches between Balance and Walk ONNX sessions
based on the commanded velocity magnitude.

Physics integration: The policy installs a ``_WBCActionController``
(with ``owns_stepping = True``) on the MuJoCo world's backend state.
This controller applies PD torques and steps physics at the model's
native dt (0.005s) with control_decimation=4, matching the NVLabs
sim2mujoco reference exactly. The arm joints (beyond the 15 policy-
controlled DOFs) are stabilized with a simple PD to zero.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)

# Default WBC configuration matching NVLabs g1_gear_wbc.yaml
_DEFAULT_CONFIG = {
    "simulation_dt": 0.005,
    "control_decimation": 4,
    "kps": [
        150.0,
        150.0,
        150.0,
        200.0,
        40.0,
        40.0,
        150.0,
        150.0,
        150.0,
        200.0,
        40.0,
        40.0,
        250.0,
        250.0,
        250.0,
    ],
    "kds": [
        2.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        5.0,
        5.0,
        5.0,
    ],
    "default_angles": [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "ang_vel_scale": 0.5,
    "dof_pos_scale": 1.0,
    "dof_vel_scale": 0.05,
    "action_scale": 0.25,
    "cmd_scale": [2.0, 2.0, 0.5],
    "num_actions": 15,
    "num_obs": 516,
    "obs_history_len": 6,
    "height_cmd": 0.74,
    "rpy_cmd": [0.0, 0.0, 0.0],
    "freq_cmd": 0.75,
    "initial_height": 0.793,
}

# 15 controlled joints (legs 12 + waist 3), ordered as in the ONNX model
_CONTROLLED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q (wxyz convention)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Conjugate quaternion
    wc, xc, yc, zc = w, -x, -y, -z
    return np.array(
        [
            v[0] * (wc**2 + xc**2 - yc**2 - zc**2) + v[1] * 2 * (xc * yc - wc * zc) + v[2] * 2 * (xc * zc + wc * yc),
            v[0] * 2 * (xc * yc + wc * zc) + v[1] * (wc**2 - xc**2 + yc**2 - zc**2) + v[2] * 2 * (yc * zc - wc * xc),
            v[0] * 2 * (xc * zc - wc * yc) + v[1] * 2 * (yc * zc + wc * xc) + v[2] * (wc**2 - xc**2 - yc**2 + zc**2),
        ],
        dtype=np.float32,
    )


def _download_onnx_models(checkpoint: str) -> tuple[Path, Path]:
    """Locate or download Balance and Walk ONNX models.

    Resolution order:
    1. Local path: if ``checkpoint`` is a directory containing the ONNX files.
    2. Cache: check ~/.cache/strands_robots/wbc/ for previously downloaded models.
    3. HuggingFace Hub: download from ``checkpoint`` as a HF repo ID.
    4. GitHub clone: shallow-clone NVlabs/GR00T-WholeBodyControl if HF fails.

    Returns (balance_path, walk_path).
    """
    balance_name = "GR00T-WholeBodyControl-Balance.onnx"
    walk_name = "GR00T-WholeBodyControl-Walk.onnx"
    onnx_subpath = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"

    # 1. Direct local path
    if Path(checkpoint).is_dir():
        policy_dir = Path(checkpoint) / onnx_subpath
        if not policy_dir.exists():
            policy_dir = Path(checkpoint)
        balance = policy_dir / balance_name
        walk = policy_dir / walk_name
        if balance.exists() and walk.exists():
            logger.info("WBC ONNX from local path: %s", policy_dir)
            return balance, walk

    # 2. Check cache directory
    cache_dir = Path.home() / ".cache" / "strands_robots" / "wbc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_balance = cache_dir / balance_name
    cached_walk = cache_dir / walk_name
    if cached_balance.exists() and cached_walk.exists():
        logger.info("WBC ONNX from cache: %s", cache_dir)
        return cached_balance, cached_walk

    # 2b. Check known local clone locations (common on Thor)
    for local_root in [
        Path("/tmp/g1-wbc-test/GR00T-WholeBodyControl"),
        Path.home() / "GR00T-WholeBodyControl",
    ]:
        policy_dir = local_root / onnx_subpath
        if (policy_dir / balance_name).exists() and (policy_dir / walk_name).exists():
            logger.info("WBC ONNX from local clone: %s", policy_dir)
            # Copy to cache for future use
            import shutil

            shutil.copy2(policy_dir / balance_name, cached_balance)
            shutil.copy2(policy_dir / walk_name, cached_walk)
            return cached_balance, cached_walk

    # 3. Try HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download

        balance_path = Path(
            hf_hub_download(
                repo_id=checkpoint,
                filename=f"{onnx_subpath}/{balance_name}",
                cache_dir=str(cache_dir),
            )
        )
        walk_path = Path(
            hf_hub_download(
                repo_id=checkpoint,
                filename=f"{onnx_subpath}/{walk_name}",
                cache_dir=str(cache_dir),
            )
        )
        logger.info("WBC ONNX from HuggingFace: %s", checkpoint)
        return balance_path, walk_path
    except Exception as hf_err:
        logger.info("HuggingFace download failed (%s), trying GitHub clone...", hf_err)

    # 4. GitHub shallow clone as fallback
    import shutil
    import subprocess
    import tempfile

    github_url = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"
    clone_dir = Path(tempfile.mkdtemp(prefix="wbc_clone_"))
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", github_url, str(clone_dir)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", onnx_subpath],
            cwd=str(clone_dir),
            check=True,
            capture_output=True,
            timeout=30,
        )
        src_balance = clone_dir / onnx_subpath / balance_name
        src_walk = clone_dir / onnx_subpath / walk_name
        if src_balance.exists() and src_walk.exists():
            shutil.copy2(src_balance, cached_balance)
            shutil.copy2(src_walk, cached_walk)
            shutil.rmtree(clone_dir, ignore_errors=True)
            logger.info("WBC ONNX from GitHub clone -> cached at %s", cache_dir)
            return cached_balance, cached_walk
    except Exception as git_err:
        logger.warning("GitHub clone failed: %s", git_err)
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)

    raise FileNotFoundError(
        f"Could not locate WBC ONNX models. Tried: "
        f"local path '{checkpoint}', cache '{cache_dir}', "
        f"HuggingFace '{checkpoint}', GitHub clone. "
        f"Please clone https://github.com/NVlabs/GR00T-WholeBodyControl "
        f"or provide the path to the ONNX files."
    )


class _WBCActionController:
    """Action controller that applies WBC torque control with its own stepping.

    Installed on the MuJoCo world's ``_backend_state["action_controller"]``.
    Mirrors the NVLabs sim2mujoco reference: PD torques for legs/waist
    (15 DOF) + arm stabilization for the remaining DOFs.
    """

    owns_stepping: bool = True

    def __init__(
        self,
        policy: WBCPolicy,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.config = config
        self.num_actions = config["num_actions"]
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.control_decimation = config["control_decimation"]
        self.physics_substeps_per_control = self.control_decimation

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,
    ) -> None:
        """Apply one control step: PD torques + physics substeps.

        ``action_dict`` carries the target DOF positions computed by the
        policy (``_target_dof_pos`` key). The controller reads it, applies
        PD torques, steps physics for ``control_decimation`` substeps, then
        queries the policy for the next ONNX inference.
        """
        import mujoco

        target_dof_pos = action_dict.get("_target_dof_pos")
        if target_dof_pos is None:
            # Fallback: use default angles
            target_dof_pos = self.default_angles.copy()

        n_joints = data.qpos.shape[0] - 7

        for _ in range(self.control_decimation):
            # PD torque for controlled joints (legs + waist)
            leg_tau = (target_dof_pos - data.qpos[7 : 7 + self.num_actions]) * self.kps - data.qvel[
                6 : 6 + self.num_actions
            ] * self.kds
            data.ctrl[: self.num_actions] = leg_tau

            # Arm stabilization: PD to zero for remaining joints
            if n_joints > self.num_actions:
                arm_tau = (
                    -data.qpos[7 + self.num_actions : 7 + n_joints] * 100.0
                    - data.qvel[6 + self.num_actions : 6 + n_joints] * 0.5
                )
                data.ctrl[self.num_actions : n_joints] = arm_tau

            mujoco.mj_step(model, data)


class WBCPolicy(Policy):
    """NVIDIA GR00T Whole-Body Control policy for humanoid locomotion.

    Runs real ONNX inference using Balance/Walk checkpoints from
    ``nvidia/GR00T-WholeBodyControl``. Maintains a 6-frame observation
    history and auto-switches policies based on commanded velocity.

    This policy sets ``requires_images = False`` (no cameras needed)
    and uses an action controller with ``owns_stepping = True`` for
    direct torque control of MuJoCo motor actuators.

    Args:
        checkpoint: HuggingFace repo ID for the ONNX models.
            Default: ``nvidia/GR00T-WholeBodyControl``.
        target_velocity: 3-element [vx, vy, wz] locomotion command.
            Default: ``[0, 0, 0]`` (balance in place).
        config: Override default WBC configuration parameters.
            See ``_DEFAULT_CONFIG`` for available keys.
    """

    def __init__(
        self,
        checkpoint: str = "nvidia/GR00T-WholeBodyControl",
        target_velocity: list[float] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._checkpoint = checkpoint
        self._target_velocity = np.array(target_velocity or [0.0, 0.0, 0.0], dtype=np.float32)
        self._config = {**_DEFAULT_CONFIG, **(config or {})}
        self._robot_state_keys: list[str] = []

        # Lazy-loaded ONNX sessions
        self._balance_session: Any | None = None
        self._walk_session: Any | None = None
        self._balance_input_name: str = ""
        self._walk_input_name: str = ""
        self._loaded = False

        # Observation history
        self._single_obs_dim = 86
        self._obs_history: collections.deque[np.ndarray] = collections.deque(
            [np.zeros(self._single_obs_dim, dtype=np.float32)] * self._config["obs_history_len"],
            maxlen=self._config["obs_history_len"],
        )
        self._obs_full = np.zeros(self._config["num_obs"], dtype=np.float32)

        # Last action for observation encoding
        self._last_action = np.zeros(self._config["num_actions"], dtype=np.float32)
        self._target_dof_pos = np.array(self._config["default_angles"], dtype=np.float32)

        # Action controller (created on first get_actions call)
        self._action_controller: _WBCActionController | None = None

        # Track whether we've initialized the robot pose
        self._initialized = False

    def _ensure_loaded(self) -> None:
        """Lazy-load ONNX models on first inference call."""
        if self._loaded:
            return

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("onnxruntime is required for WBC policy. Install with: pip install onnxruntime") from e

        balance_path, walk_path = _download_onnx_models(self._checkpoint)

        self._balance_session = ort.InferenceSession(str(balance_path), providers=["CPUExecutionProvider"])
        self._walk_session = ort.InferenceSession(str(walk_path), providers=["CPUExecutionProvider"])
        self._balance_input_name = self._balance_session.get_inputs()[0].name
        self._walk_input_name = self._walk_session.get_inputs()[0].name

        self._loaded = True
        logger.info(
            "WBC ONNX sessions loaded (checkpoint=%s, balance_input=%s, walk_input=%s)",
            self._checkpoint,
            self._balance_input_name,
            self._walk_input_name,
        )

    def _compute_single_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        n_joints: int,
    ) -> np.ndarray:
        """Compute a single 86-dim observation from MuJoCo state.

        Observation layout (matching NVLabs upstream):
            [ 0: 7] command = [vx*2, vy*2, wz*0.5, height, roll, pitch, yaw]
            [ 7:10] base_omega * 0.5
            [10:13] gravity_orientation = quat_rotate_inverse(base_quat, [0,0,-1])
            [13:42] (qpos[7:36] - default_angles_padded) * dof_pos_scale
            [42:71] qvel[6:35] * dof_vel_scale
            [71:86] prev_action (15-dim)
        """
        cfg = self._config
        cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

        command = np.zeros(7, dtype=np.float32)
        command[:3] = self._target_velocity * cmd_scale
        command[3] = cfg["height_cmd"]
        command[4:7] = np.array(cfg["rpy_cmd"], dtype=np.float32)

        quat = qpos[3:7].copy()
        omega = qvel[3:6].copy()

        # Padded defaults for all joints (policy only controls 15 but obs covers all)
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        n_default = min(len(cfg["default_angles"]), n_joints)
        padded_defaults[:n_default] = np.array(cfg["default_angles"][:n_default], dtype=np.float32)

        qj_scaled = (qpos[7 : 7 + n_joints] - padded_defaults) * cfg["dof_pos_scale"]
        dqj_scaled = qvel[6 : 6 + n_joints] * cfg["dof_vel_scale"]
        gravity_orientation = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        omega_scaled = omega * cfg["ang_vel_scale"]

        obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        obs[0:7] = command[:7]
        obs[7:10] = omega_scaled
        obs[10:13] = gravity_orientation
        obs[13 : 13 + n_joints] = qj_scaled
        obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        obs[13 + 2 * n_joints : 13 + 2 * n_joints + self._config["num_actions"]] = self._last_action

        return obs

    def _run_inference(self) -> np.ndarray:
        """Run ONNX inference on the current observation history.

        Auto-switches between Balance (||cmd|| <= 0.05) and Walk sessions.
        Returns 15-dim raw action output.
        """
        # Build full observation from history
        for i, obs_frame in enumerate(self._obs_history):
            self._obs_full[i * self._single_obs_dim : (i + 1) * self._single_obs_dim] = obs_frame

        obs_input = self._obs_full[None, :].astype(np.float32)

        cmd_norm = float(np.linalg.norm(self._target_velocity))
        if cmd_norm <= 0.05:
            result = self._balance_session.run(None, {self._balance_input_name: obs_input})  # type: ignore[union-attr]
        else:
            result = self._walk_session.run(None, {self._walk_input_name: obs_input})  # type: ignore[union-attr]

        return result[0].squeeze().astype(np.float32)

    @property
    def provider_name(self) -> str:
        return "wbc"

    @property
    def requires_images(self) -> bool:
        """WBC only needs proprioceptive state - no cameras."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._robot_state_keys = robot_state_keys

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state (observation history, actions)."""
        self._obs_history = collections.deque(
            [np.zeros(self._single_obs_dim, dtype=np.float32)] * self._config["obs_history_len"],
            maxlen=self._config["obs_history_len"],
        )
        self._obs_full = np.zeros(self._config["num_obs"], dtype=np.float32)
        self._last_action = np.zeros(self._config["num_actions"], dtype=np.float32)
        self._target_dof_pos = np.array(self._config["default_angles"], dtype=np.float32)
        self._initialized = False

    def set_target_velocity(self, velocity: list[float]) -> None:
        """Update the locomotion command mid-episode.

        Args:
            velocity: [vx, vy, wz] in m/s and rad/s.
        """
        self._target_velocity = np.array(velocity, dtype=np.float32)

    def get_action_controller(self) -> _WBCActionController:
        """Get or create the action controller for MuJoCo integration.

        The controller is installed on the world's backend_state by the
        simulation layer when it detects a WBC policy.
        """
        if self._action_controller is None:
            self._action_controller = _WBCActionController(self, self._config)
        return self._action_controller

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Compute WBC actions from observation.

        The observation_dict should contain raw MuJoCo state under special
        keys injected by the WBC-aware observation path:
            - ``_wbc_qpos``: full qpos array (free joint + all DOFs)
            - ``_wbc_qvel``: full qvel array
            - ``_wbc_n_joints``: number of articulated joints

        If these keys are absent, the policy falls back to returning
        the current target DOF positions (useful for the first frame
        before the controller is active).

        Returns a single action dict with ``_target_dof_pos`` for the
        action controller to consume.
        """
        self._ensure_loaded()

        qpos = observation_dict.get("_wbc_qpos")
        qvel = observation_dict.get("_wbc_qvel")
        n_joints = observation_dict.get("_wbc_n_joints")

        if qpos is not None and qvel is not None and n_joints is not None:
            # Compute observation and run inference
            single_obs = self._compute_single_obs(
                np.asarray(qpos, dtype=np.float32),
                np.asarray(qvel, dtype=np.float32),
                int(n_joints),
            )
            self._obs_history.append(single_obs)

            action = self._run_inference()
            self._last_action = action.copy()
            self._target_dof_pos = action * self._config["action_scale"] + np.array(
                self._config["default_angles"], dtype=np.float32
            )

        # Return the target DOF positions for the action controller
        return [{"_target_dof_pos": self._target_dof_pos.copy()}]
