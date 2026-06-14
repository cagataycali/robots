"""WBC-specific run_policy implementation for MuJoCo.

Bypasses the standard PolicyRunner loop to run the tight WBC control
loop (dt=0.005, control_decimation=4) matching the NVLabs sim2mujoco
reference. This is necessary because WBC's physics/control timing
requirements differ from the standard 50Hz policy-query loop.

The runner:
1. Initializes the robot in a standing pose
2. Runs the ONNX inference + PD torque loop at native physics rate
3. Optionally records video
4. Returns standard status dict with metrics
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.policies.wbc.assets import get_wbc_xml_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from strands_robots.policies.wbc.wbc_policy import WBCPolicy


def run_wbc_policy(
    sim: Any,
    robot_name: str,
    policy: WBCPolicy,
    duration: float = 10.0,
    target_velocity: list[float] | None = None,
    video: dict[str, Any] | None = None,
    fast_mode: bool = False,
    on_frame: Any | None = None,
) -> dict[str, Any]:
    """Run WBC policy loop directly on MuJoCo model/data.

    This replaces the standard PolicyRunner.run() for WBC because the
    whole-body controller requires:
    - Direct torque writes to data.ctrl (motor actuators)
    - Native physics dt (0.005s) with control_decimation=4
    - Raw qpos/qvel observation (not joint-name dict)

    Args:
        sim: MuJoCo Simulation instance (has _world, _lock).
        robot_name: Robot name in the world.
        policy: Initialized WBCPolicy instance.
        duration: Simulation duration in seconds.
        target_velocity: [vx, vy, wz] override (or uses policy default).
        video: Video recording config dict (path, fps, width, height).
        fast_mode: Skip real-time sleep.
        on_frame: Optional on_frame hook (step, obs, action) for recording.

    Returns:
        Standard status dict with WBC-specific metrics.
    """
    import mujoco

    world = sim._world
    if world is None:
        return {"status": "error", "content": [{"text": "No world."}]}

    # Override target velocity if provided
    if target_velocity is not None:
        policy.set_target_velocity(target_velocity)

    # Ensure ONNX models are loaded
    policy._ensure_loaded()

    config = policy._config

    # Load WBC-specific XML (motor actuators, torque control).
    # The standard Menagerie G1 uses position actuators which are
    # incompatible with WBC's PD torque control loop.
    checkpoint = policy._checkpoint
    wbc_xml_path = get_wbc_xml_path(checkpoint)
    logger.info("Loading WBC XML: %s", wbc_xml_path)
    model = mujoco.MjModel.from_xml_path(str(wbc_xml_path))
    data = mujoco.MjData(model)

    # Install the WBC model/data on the world so other sim methods
    # (get_robot_pose, render, etc.) see the correct state.
    world._model = model
    world._data = data

    n_joints = data.qpos.shape[0] - 7
    num_actions = config["num_actions"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    # Set physics timestep to match WBC requirements
    model.opt.timestep = simulation_dt

    # Initialize robot in standing pose
    if not policy._initialized:
        data.qpos[2] = config["initial_height"]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright quaternion
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        data.qpos[7 : 7 + num_actions] = default_angles
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        policy._initialized = True
        logger.info(
            "WBC initialized: height=%.3f, n_joints=%d, num_actions=%d",
            config["initial_height"],
            n_joints,
            num_actions,
        )

    # PD gains
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    # State
    target_dof_pos = np.array(config["default_angles"], dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)

    n_steps = int(duration / simulation_dt)
    step_counter = 0
    policy_steps = 0

    # Video recording setup
    writer = None
    renderer = None
    video_fps = 30
    render_every = max(1, int((1.0 / video_fps) / simulation_dt))

    if video and video.get("path"):
        try:
            import cv2

            video_path = video["path"]
            video_fps = video.get("fps", 30)
            width = video.get("width", 640)
            height = video.get("height", 480)
            render_every = max(1, int((1.0 / video_fps) / simulation_dt))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
            renderer = mujoco.Renderer(model, height=height, width=width)
            logger.info("WBC video recording: %s (%dx%d @ %dfps)", video_path, width, height, video_fps)
        except Exception as e:
            logger.warning("WBC video setup failed: %s", e)
            writer = None

    # Tracking camera setup for video
    cam = None
    if renderer is not None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id >= 0:
            cam.trackbodyid = pelvis_id
        cam.distance = 3.0
        cam.elevation = -15
        cam.azimuth = 135

    # Metrics
    min_height = float("inf")
    initial_x = float(data.qpos[0])
    t0 = time.time()

    try:
        for step in range(n_steps):
            # PD torque for controlled joints
            leg_tau = (target_dof_pos - data.qpos[7 : 7 + num_actions]) * kps - data.qvel[6 : 6 + num_actions] * kds
            data.ctrl[:num_actions] = leg_tau

            # Arm stabilization
            if n_joints > num_actions:
                arm_tau = (
                    -data.qpos[7 + num_actions : 7 + n_joints] * 100.0 - data.qvel[6 + num_actions : 6 + n_joints] * 0.5
                )
                data.ctrl[num_actions:n_joints] = arm_tau

            mujoco.mj_step(model, data)
            step_counter += 1
            min_height = min(min_height, float(data.qpos[2]))

            # Policy inference at control rate
            if step_counter % control_decimation == 0:
                # Compute observation
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                single_obs = policy._compute_single_obs(qpos, qvel, n_joints)
                policy._obs_history.append(single_obs)

                # Run ONNX inference
                action = policy._run_inference()
                policy._last_action = action.copy()
                target_dof_pos = action * config["action_scale"] + np.array(config["default_angles"], dtype=np.float32)
                policy._target_dof_pos = target_dof_pos.copy()
                policy_steps += 1

                # on_frame callback (for recording hooks)
                if on_frame is not None:
                    try:
                        obs_dict = {
                            "_wbc_qpos": qpos,
                            "_wbc_qvel": qvel,
                            "_wbc_n_joints": n_joints,
                        }
                        action_dict = {"_target_dof_pos": target_dof_pos}
                        on_frame(policy_steps, obs_dict, action_dict)
                    except Exception:
                        # CooperativeStop or similar - propagate
                        raise

            # Video frame
            if writer is not None and step % render_every == 0:
                try:
                    assert renderer is not None
                    renderer.update_scene(data, camera=cam)
                    frame = renderer.render()
                    import cv2

                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    logger.debug("Video frame write failed: %s", e)

    except Exception as e:
        # Check for cooperative stop (from on_frame hook)
        if "CooperativeStop" in type(e).__name__ or "PolicyStopped" in type(e).__name__:
            logger.info("WBC policy stopped cooperatively")
        else:
            raise
    finally:
        if writer is not None:
            writer.release()
        # Update world state
        if world is not None:
            world.sim_time = data.time
            world.step_count = getattr(world, "step_count", 0) + step_counter

    elapsed = time.time() - t0
    final_height = float(data.qpos[2])
    final_x = float(data.qpos[0])
    fell = final_height < 0.5
    distance = final_x - initial_x

    metrics = {
        "final_height": final_height,
        "min_height": min_height,
        "distance_x": distance,
        "fell": fell,
        "policy_steps": policy_steps,
        "physics_steps": step_counter,
        "wall_time": elapsed,
        "sim_time": float(data.time),
    }

    status_text = (
        f"WBC completed: {duration}s sim in {elapsed:.1f}s wall, "
        f"height={final_height:.3f}m (min={min_height:.3f}m), "
        f"distance_x={distance:.2f}m, fell={fell}"
    )
    logger.info(status_text)

    return {
        "status": "success",
        "content": [{"text": status_text}],
        "metrics": metrics,
    }
