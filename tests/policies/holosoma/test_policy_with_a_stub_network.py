"""HolosomaPolicy with a stub network: contract, refusals, action mapping, both observation shapes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.holosoma import (
    HOLOSOMA_FILES,
    HOLOSOMA_G1_JOINTS,
    HOLOSOMA_HF_REPO,
    HolosomaConfig,
    HolosomaPolicy,
    resolve_holosoma_checkpoint,
)
from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, PDTorquePolicy

KP = tuple(float(40 + i) for i in range(29))
KD = tuple(float(1 + i / 10) for i in range(29))


class _StubSession:
    """Records the observation it was fed and answers a fixed 29-wide action."""

    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = np.zeros(29) if action is None else np.asarray(action, dtype=np.float32)
        self.seen: list[np.ndarray] = []

    def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert list(feed) == ["actor_obs"]
        obs = feed["actor_obs"]
        assert obs.shape == (1, 100) and obs.dtype == np.float32
        self.seen.append(obs.copy())
        return [self.action.reshape(1, -1)]


def _policy(action: np.ndarray | None = None, **kwargs: Any) -> tuple[HolosomaPolicy, _StubSession]:
    pol = HolosomaPolicy(allow_missing_models=True, config=HolosomaConfig(kps=KP, kds=KD), **kwargs)
    stub = _StubSession(action)
    pol.session = stub
    return pol, stub


def _sim_obs(q: np.ndarray | None = None) -> dict[str, Any]:
    q = np.zeros(29) if q is None else q
    obs: dict[str, Any] = {name: float(v) for name, v in zip(HOLOSOMA_G1_JOINTS, q, strict=True)}
    obs.update({f"{name}.vel": 0.0 for name in HOLOSOMA_G1_JOINTS})
    obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    obs["base_ang_vel"] = [0.0, 0.0, 0.0]
    return obs


def test_contract_flags() -> None:
    pol, _ = _policy()
    assert pol.provider_name == "holosoma"
    assert pol.requires_images is False
    assert HolosomaPolicy.reads_instruction is False
    assert HolosomaPolicy.pd_torque_shim is True
    assert HolosomaPolicy.requires_action_controller and "mujoco" in HolosomaPolicy.requires_action_controller
    assert isinstance(pol, PDTorquePolicy)
    assert HOLOSOMA_G1_JOINTS == WBC_G1_ALL_JOINTS
    assert pol.config.num_actions == pol.config.n_obs_joints == 29


def test_action_is_default_plus_quarter_of_the_clipped_output_for_all_29_joints() -> None:
    raw = np.full(29, 2.0)
    raw[0] = 400.0  # clipped to 100
    pol, _ = _policy(raw)
    pol.set_robot_state_keys(["floating_base_joint", *HOLOSOMA_G1_JOINTS])
    actions = asyncio.run(pol.get_actions(_sim_obs(), "ignored", target_velocity=[0.5, 0.0, 0.0]))
    assert len(actions) == 1
    act = actions[0]
    assert list(act) == list(HOLOSOMA_G1_JOINTS)
    default = np.asarray(pol.default_angles)
    assert act["left_hip_pitch_joint"] == pytest.approx(default[0] + 0.25 * 100.0)
    assert act["right_wrist_yaw_joint"] == pytest.approx(default[28] + 0.25 * 2.0)
    # the clipped raw action is what the next observation sees (upstream last_policy_action)
    asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.5, 0.0, 0.0]))
    fed = pol.session.seen[1][0]
    assert fed[0] == pytest.approx(100.0)
    assert fed[1] == pytest.approx(2.0)


def test_observation_is_fed_in_the_alphabetical_layout() -> None:
    pol, stub = _policy()
    q = np.asarray(pol.default_angles) + 0.1
    obs = _sim_obs(q)
    obs["base_ang_vel"] = [0.4, 0.0, 0.0]
    obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    asyncio.run(pol.get_actions(obs, "", target_velocity=[0.5, -0.2, 0.3]))
    fed = stub.seen[0][0]
    np.testing.assert_allclose(fed[0:29], 0.0)  # no previous action
    np.testing.assert_allclose(fed[29:32], [0.1, 0.0, 0.0])  # 0.4 * 0.25
    assert fed[32] == pytest.approx(0.3)
    np.testing.assert_allclose(fed[33:35], [0.5, -0.2])
    dt = pol.config.phase_dt
    np.testing.assert_allclose(fed[35:37], np.cos([dt, -np.pi + dt]), atol=1e-6)  # phase advanced before obs
    np.testing.assert_allclose(fed[37:66], 0.1, atol=1e-6)
    np.testing.assert_allclose(fed[66:95], 0.0)
    np.testing.assert_allclose(fed[95:98], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(fed[98:100], np.sin([dt, -np.pi + dt]), atol=1e-6)


def test_hardware_snapshot_shape_is_read_like_the_sim_shape() -> None:
    q = np.linspace(-0.3, 0.3, 29)
    pol_sim, stub_sim = _policy()
    pol_hw, stub_hw = _policy()
    sim = _sim_obs(q)
    sim["base_ang_vel"] = [0.1, 0.2, 0.3]
    hw = {
        "joints": {name: {"q": float(v), "dq": 0.0} for name, v in zip(HOLOSOMA_G1_JOINTS, q, strict=True)},
        "imu": {"quaternion": [1.0, 0.0, 0.0, 0.0], "gyroscope": [0.1, 0.2, 0.3]},
    }
    asyncio.run(pol_sim.get_actions(sim, "", target_velocity=[0.3, 0.0, 0.0]))
    asyncio.run(pol_hw.get_actions(hw, "", target_velocity=[0.3, 0.0, 0.0]))
    np.testing.assert_allclose(stub_sim.seen[0], stub_hw.seen[0])


def test_arm_observation_default_hides_the_measured_arms() -> None:
    q = np.asarray(HolosomaConfig().default_angles) + 0.5
    pol, stub = _policy(arm_observation="default")
    asyncio.run(pol.get_actions(_sim_obs(q), "", target_velocity=[0.3, 0.0, 0.0]))
    fed = stub.seen[0][0]
    np.testing.assert_allclose(fed[37:52], 0.5, atol=1e-6)  # legs + waist: measured
    np.testing.assert_allclose(fed[52:66], 0.0, atol=1e-6)  # arms: default


def test_driven_joints_legs_waist_emits_fifteen_targets() -> None:
    pol, _ = _policy(driven_joints="legs_waist")
    act = asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.3, 0.0, 0.0]))[0]
    assert list(act) == list(HOLOSOMA_G1_JOINTS[:15])


def test_zero_command_is_the_stand_posture_and_constructor_default_applies() -> None:
    pol, stub = _policy(target_velocity=[0.4, 0.0, 0.0])
    asyncio.run(pol.get_actions(_sim_obs(), ""))
    assert stub.seen[0][0][33] == pytest.approx(0.4)
    pol2, stub2 = _policy()
    asyncio.run(pol2.get_actions(_sim_obs(), ""))
    fed = stub2.seen[0][0]
    np.testing.assert_allclose(fed[32:35], 0.0)
    np.testing.assert_allclose(fed[35:37], -1.0, atol=1e-6)  # cos(pi) both feet: standing


def test_command_is_clipped_to_the_checkpoint_ranges() -> None:
    pol, stub = _policy()
    pol.apply_metadata({"command_ranges": {"lin_vel_x": [-0.6, 0.6], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 1]}})
    asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.9, 0.0, 0.0]))
    assert stub.seen[0][0][33] == pytest.approx(0.6)


def test_metadata_gains_fill_the_config_and_config_gains_win() -> None:
    pol = HolosomaPolicy(allow_missing_models=True)
    assert pol.kps is None
    with pytest.raises(RuntimeError, match="no PD gains"):
        pol.compute_torques(np.zeros(29), np.zeros(29), np.zeros(29))
    pol.apply_metadata({"kp": list(KP), "kd": list(KD), "dof_names": list(HOLOSOMA_G1_JOINTS)})
    np.testing.assert_allclose(pol.kps, KP)
    tau = pol.compute_torques(np.ones(29), np.zeros(29), np.full(29, 2.0))
    np.testing.assert_allclose(tau, np.asarray(KP) - 2.0 * np.asarray(KD))
    override = HolosomaPolicy(allow_missing_models=True, config=HolosomaConfig(kps=KD, kds=KP))
    override.apply_metadata({"kp": list(KP), "kd": list(KD)})
    np.testing.assert_allclose(override.kps, KD)


def test_metadata_with_foreign_joint_names_is_refused() -> None:
    pol = HolosomaPolicy(allow_missing_models=True)
    with pytest.raises(RuntimeError, match="dof_names"):
        pol.apply_metadata({"dof_names": ["AAHead_yaw", "Head_pitch"]})


def test_non_g1_joint_list_is_refused_by_name() -> None:
    pol, _ = _policy()
    with pytest.raises(ValueError, match="missing expected Unitree G1 joints") as exc:
        pol.set_robot_state_keys(["shoulder_pan", "elbow_flex"])
    assert "left_hip_pitch_joint" in str(exc.value)


def test_missing_session_refusal_names_the_seam() -> None:
    pol = HolosomaPolicy(allow_missing_models=True)
    with pytest.raises(RuntimeError, match="policy.session"):
        asyncio.run(pol.get_actions(_sim_obs(), ""))


def test_non_finite_network_output_is_refused() -> None:
    pol, _ = _policy(np.full(29, np.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.3, 0.0, 0.0]))


@pytest.mark.parametrize("bad", [[0.5, 0.0], [float("nan"), 0.0, 0.0], "fast"])
def test_target_velocity_domain(bad: Any) -> None:
    pol, _ = _policy()
    with pytest.raises(ValueError):
        asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=bad))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"allow_missing_models": "false"}, "allow_missing_models"),
        ({"driven_joints": "arms"}, "driven_joints"),
        ({"arm_observation": "hidden"}, "arm_observation"),
        ({"algorithm": "sac"}, "algorithm"),
    ],
)
def test_constructor_flag_domains(kwargs: dict[str, Any], match: str) -> None:
    base = {"allow_missing_models": True}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        HolosomaPolicy(**base)


def test_reset_clears_the_previous_action_and_the_clock() -> None:
    pol, stub = _policy(np.ones(29))
    asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.3, 0.0, 0.0]))
    pol.reset()
    asyncio.run(pol.get_actions(_sim_obs(), "", target_velocity=[0.3, 0.0, 0.0]))
    np.testing.assert_allclose(stub.seen[1][0][0:29], 0.0)
    dt = pol.config.phase_dt
    np.testing.assert_allclose(stub.seen[1][0][98:100], np.sin([dt, -np.pi + dt]), atol=1e-6)


def test_checkpoint_resolution_paths(tmp_path: Path) -> None:
    onnx = tmp_path / HOLOSOMA_FILES["ppo"]
    onnx.write_bytes(b"x")
    assert resolve_holosoma_checkpoint(onnx, "fastsac") == onnx
    assert resolve_holosoma_checkpoint(tmp_path, "ppo") == onnx
    with pytest.raises(FileNotFoundError, match="has no 'fastsac_g1_29dof.onnx'"):
        resolve_holosoma_checkpoint(tmp_path, "fastsac")
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_holosoma_checkpoint(tmp_path / "nope" / "x.onnx", "fastsac")
    assert HOLOSOMA_HF_REPO == "nepyope/holosoma_locomotion"


def test_config_domains() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        HolosomaConfig(num_actions=15)
    with pytest.raises(ValueError, match="default_angles"):
        HolosomaConfig(default_angles=(0.0,) * 15)
    with pytest.raises(ValueError, match="kps"):
        HolosomaConfig(kps=(1.0,) * 3)
    with pytest.raises(ValueError, match="gait_period"):
        HolosomaConfig(gait_period=0.0)
    cfg = HolosomaConfig()
    assert json.dumps(cfg.command_ranges)  # plain data


def test_registry_lists_the_provider() -> None:
    from strands_robots.policies import create_policy, list_providers

    assert "holosoma" in list_providers()
    pol = create_policy("holosoma", allow_missing_models=True)
    assert isinstance(pol, HolosomaPolicy)
    assert pol.config.algorithm == "fastsac"
