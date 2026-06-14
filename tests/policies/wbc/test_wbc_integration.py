"""Integration tests for WBCPolicy with real ONNX models in MuJoCo.

These tests require:
- ONNX models from NVlabs/GR00T-WholeBodyControl at a known local path
- MuJoCo with EGL rendering (MUJOCO_GL=egl)
- onnxruntime

Marked with pytest.mark.wbc so they can be skipped in CI without GPU/models.
Run with: MUJOCO_GL=egl pytest -m wbc tests/policies/wbc/test_wbc_integration.py
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

# Skip entire module if ONNX models or MuJoCo are unavailable
WBC_BASE = Path("/tmp/g1-wbc-test/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1")
BALANCE_ONNX = WBC_BASE / "policy" / "GR00T-WholeBodyControl-Balance.onnx"
WALK_ONNX = WBC_BASE / "policy" / "GR00T-WholeBodyControl-Walk.onnx"
G1_XML = WBC_BASE / "g1_gear_wbc.xml"

_MODELS_AVAILABLE = BALANCE_ONNX.exists() and WALK_ONNX.exists() and G1_XML.exists()
_SKIP_REASON = "WBC ONNX models not available locally (need NVlabs/GR00T-WholeBodyControl clone)"

pytestmark = [
    pytest.mark.wbc,
    pytest.mark.skipif(not _MODELS_AVAILABLE, reason=_SKIP_REASON),
]

# Config from verified g1_gear_wbc.yaml
SIM_DT = 0.005
CONTROL_DECIMATION = 4
NUM_ACTIONS = 15


def _run_sim(variant: str, target_velocity: list, duration: float = 5.0):
    """Run WBC policy in MuJoCo with PD torque control."""
    import mujoco

    from strands_robots.policies.wbc import WBC_JOINT_NAMES, WBCPolicy

    policy = WBCPolicy(
        balance_onnx=str(BALANCE_ONNX),
        walk_onnx=str(WALK_ONNX),
        variant=variant,
    )
    policy.set_robot_state_keys(WBC_JOINT_NAMES)

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    n_joints = data.qpos.shape[0] - 7

    # Initialize (matching reference)
    default_angles = np.array(
        [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    data.qpos[2] = 0.793
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7 : 7 + NUM_ACTIONS] = default_angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Actuator mapping (MuJoCo uses "_joint" suffix)
    wbc_to_actuator = {}
    for i, jname in enumerate(WBC_JOINT_NAMES):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname + "_joint")
        if act_id >= 0:
            wbc_to_actuator[i] = act_id

    n_steps = int(duration / SIM_DT)
    counter = 0
    pelvis_z_min = float("inf")
    start_x = data.qpos[0]
    target_dof_pos = default_angles.copy()

    for step in range(n_steps):
        # PD torque control
        current_pos = data.qpos[7 : 7 + NUM_ACTIONS].copy()
        current_vel = data.qvel[6 : 6 + NUM_ACTIONS].copy()
        leg_tau = policy.compute_torques(target_dof_pos, current_pos, current_vel)
        for i, act_id in wbc_to_actuator.items():
            data.ctrl[act_id] = leg_tau[i]

        # Arm PD hold
        if n_joints > NUM_ACTIONS:
            arm_tau = (
                -data.qpos[7 + NUM_ACTIONS : 7 + n_joints] * 100.0 - data.qvel[6 + NUM_ACTIONS : 6 + n_joints] * 0.5
            )
            data.ctrl[NUM_ACTIONS:] = arm_tau

        mujoco.mj_step(model, data)
        counter += 1
        pelvis_z_min = min(pelvis_z_min, data.qpos[2])

        # Policy at control frequency
        if counter % CONTROL_DECIMATION == 0:
            obs_dict = {
                "observation.state": data.qpos[7 : 7 + n_joints].copy(),
                "observation.velocity": data.qvel[6 : 6 + n_joints].copy(),
                "observation.base_quat": data.qpos[3:7].copy(),
                "observation.base_angular_velocity": data.qvel[3:6].copy(),
            }
            actions = asyncio.run(
                policy.get_actions(
                    obs_dict, "", target_velocity=target_velocity, target_height=0.74, target_rpy=[0.0, 0.0, 0.0]
                )
            )
            action_dict = actions[0]
            for i, jname in enumerate(WBC_JOINT_NAMES):
                if jname in action_dict:
                    target_dof_pos[i] = action_dict[jname]

    return pelvis_z_min, data.qpos[2], data.qpos[0] - start_x


class TestWBCBalanceIntegration:
    """G1 balances in place with real ONNX Balance model."""

    def test_stands_upright_5s(self):
        z_min, z_final, _ = _run_sim("balance", [0.0, 0.0, 0.0], duration=5.0)
        assert z_min > 0.5, f"Robot fell: min Z = {z_min:.4f}m"
        assert z_final > 0.7, f"Robot sagging: final Z = {z_final:.4f}m"

    def test_pelvis_height_stable(self):
        z_min, z_final, _ = _run_sim("balance", [0.0, 0.0, 0.0], duration=5.0)
        assert abs(z_final - 0.74) < 0.05, f"Height drift: {z_final:.4f}m (target 0.74m)"


class TestWBCWalkIntegration:
    """G1 walks forward with real ONNX Walk model."""

    def test_walks_forward(self):
        z_min, z_final, dx = _run_sim("walk", [0.5, 0.0, 0.0], duration=5.0)
        assert z_min > 0.5, f"Robot fell while walking: min Z = {z_min:.4f}m"
        assert dx > 1.0, f"Insufficient forward progress: {dx:.2f}m (expected >1m in 5s)"

    def test_stays_upright_while_walking(self):
        z_min, z_final, _ = _run_sim("walk", [0.5, 0.0, 0.0], duration=5.0)
        assert z_final > 0.6, f"Robot ended low: {z_final:.4f}m"


class TestWBCTurnIntegration:
    """G1 turns in place with real ONNX Walk model."""

    def test_turns_without_falling(self):
        z_min, z_final, _ = _run_sim("walk", [0.0, 0.0, 0.5], duration=5.0)
        assert z_min > 0.5, f"Robot fell while turning: min Z = {z_min:.4f}m"
