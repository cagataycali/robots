"""An exported Isaac Lab actor reads a strands robot's own state and drives its torque motors as Isaac Lab would.

The deploy contract (1/2) says what the actor's outputs mean. This half closes
the loop on a strands robot:

* the input vector is built from the strands observation term by term, in
  Isaac Lab's frames: strands reports ``base_quat`` as ``[w, x, y, z]`` and
  ``base_lin_vel`` in the world frame, Isaac Lab reads body-frame velocities
  and gravity projected into the body; ``joint_pos_rel`` is relative to the
  run's default pose and ``last_action`` is the actor's previous output;
* the run's ``params/env.yaml`` names the term the IO descriptors skip (the
  rough-terrain Go2's 187-value ``height_scan``) and the actuator model
  (a ``DCMotor`` PD, stiffness 25, damping 0.5);
* on MuJoCo torque motors that PD runs every physics step. Measured with the
  published Go2 rough-terrain policy on strands' MuJoCo Go2: standing, the base
  settles at 0.312 m, upright 0.978; without the PD it collapses to 0.077 m.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from strands_robots.training.rl.deploy_contract import (
    DeployContractError,
    attach_env_cfg,
    build_policy_obs,
    complete_obs_layout,
    contract_from_io_descriptors,
    env_cfg_actuators,
)
from tests.training.test_isaaclab_deploy_contract import _GO2, go2_io_descriptors

_GO2_ENV_CFG: dict[str, Any] = {
    "observations": {
        "policy": {
            "concatenate_terms": True,
            "base_lin_vel": {"func": "isaaclab.envs.mdp.observations:base_lin_vel", "clip": None, "scale": None},
            "base_ang_vel": {"func": "isaaclab.envs.mdp.observations:base_ang_vel"},
            "projected_gravity": {"func": "isaaclab.envs.mdp.observations:projected_gravity"},
            "velocity_commands": {
                "func": "isaaclab.envs.mdp.observations:generated_commands",
                "params": {"command_name": "base_velocity"},
            },  # fmt: skip
            "joint_pos": {"func": "isaaclab.envs.mdp.observations:joint_pos_rel"},
            "joint_vel": {"func": "isaaclab.envs.mdp.observations:joint_vel_rel"},
            "actions": {"func": "isaaclab.envs.mdp.observations:last_action"},
            "height_scan": {"func": "isaaclab.envs.mdp.observations:height_scan", "clip": [-1.0, 1.0]},
        }
    },
    "scene": {
        "robot": {
            "actuators": {
                "base_legs": {
                    "class_type": "isaaclab.actuators.actuator_pd:DCMotor",
                    "joint_names_expr": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                    "stiffness": 25.0,
                    "damping": 0.5,
                    "saturation_effort": {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
                    "effort_limit": None,
                }
            }
        }
    },
}


def _rough_contract() -> dict[str, Any]:
    contract = attach_env_cfg(contract_from_io_descriptors(go2_io_descriptors()), _GO2_ENV_CFG)
    return complete_obs_layout(contract, num_actor_obs=235)


def _standing_obs(yaw: float = 0.0) -> dict[str, Any]:
    obs: dict[str, Any] = {j: 0.0 for j in _GO2}
    obs.update({f"{j}.vel": 0.0 for j in _GO2})
    obs.update(
        base_pos=[0.0, 0.0, 0.35],
        base_quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        base_lin_vel=[1.0, 0.0, 0.0],
        base_ang_vel=[0.0, 0.0, 0.2],
    )
    return obs


class TestTheRunsEnvConfigCompletesTheContract:
    def test_the_undescribed_height_scan_is_named_and_placed_last(self) -> None:
        contract = _rough_contract()
        assert contract["obs_layout_complete"] and contract["num_obs"] == 235
        last = contract["obs_layout"][-1]
        assert (last["term"], last["func"], last["start"], last["width"]) == ("height_scan", "height_scan", 48, 187)
        assert [t["term"] for t in contract["obs_layout"]][3] == "velocity_commands"

    def test_the_actuator_model_is_resolved_per_joint(self) -> None:
        actuators = env_cfg_actuators(_GO2_ENV_CFG, _GO2)
        assert actuators["FL_calf_joint"] == {
            "model": "DCMotor",
            "stiffness": 25.0,
            "damping": 0.5,
            "effort_limit": 45.43,
        }
        assert actuators["RR_hip_joint"]["effort_limit"] == 23.7

    def test_two_undescribed_terms_leave_the_layout_incomplete(self) -> None:
        env = json.loads(json.dumps(_GO2_ENV_CFG))
        env["observations"]["policy"]["contacts"] = {"func": "isaaclab.envs.mdp.observations:contact_forces"}
        contract = complete_obs_layout(
            attach_env_cfg(contract_from_io_descriptors(go2_io_descriptors()), env), num_actor_obs=235
        )
        assert not contract["obs_layout_complete"] and contract["obs_unaccounted"] == 187


class TestTheObservationIsBuiltInIsaacLabsFrames:
    def test_velocities_and_gravity_are_in_the_body_frame(self) -> None:
        contract = _rough_contract()
        keys = {j: j for j in _GO2}
        obs = build_policy_obs(contract, _standing_obs(yaw=math.pi / 2), joint_keys=keys, last_action=[0.0] * 12)
        assert obs[0:3] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)  # world +x seen from a base yawed 90 deg
        assert obs[3:6] == pytest.approx([0.0, 0.0, 0.2])  # strands reports it in the body frame already
        assert obs[6:9] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
        assert obs[9:12] == [0.0, 0.0, 0.0]

    def test_a_tilted_base_projects_gravity(self) -> None:
        contract = _rough_contract()
        obs = _standing_obs()
        obs["base_quat"] = [math.cos(0.25), math.sin(0.25), 0.0, 0.0]  # 0.5 rad roll
        built = build_policy_obs(contract, obs, joint_keys={j: j for j in _GO2}, last_action=[0.0] * 12)
        assert built[6:9] == pytest.approx([0.0, -math.sin(0.5), -math.cos(0.5)], abs=1e-9)

    def test_joints_are_relative_to_the_default_pose_and_the_height_scan_to_flat_ground(self) -> None:
        contract = _rough_contract()
        obs = _standing_obs()
        obs["RL_thigh_joint"] = 1.25
        built = build_policy_obs(
            contract, obs, joint_keys={j: j for j in _GO2}, last_action=[0.5] * 12, command=[0.5, 0, 0]
        )
        assert built[9:12] == [0.5, 0.0, 0.0]
        assert built[12 + _GO2.index("RL_thigh_joint")] == pytest.approx(0.25)
        assert built[12 + _GO2.index("FL_calf_joint")] == pytest.approx(1.5)
        assert built[36:48] == [0.5] * 12
        assert built[48:] == pytest.approx([0.35 - 0.5] * 187)

    def test_a_term_strands_cannot_compute_is_refused_unless_supplied(self) -> None:
        contract = _rough_contract()
        contract["obs_layout"][-1] = {**contract["obs_layout"][-1], "func": "contact_forces", "term": "contacts"}
        with pytest.raises(DeployContractError, match=r"'contact_forces' term \(contacts\).*obs_terms"):
            build_policy_obs(contract, _standing_obs(), joint_keys={j: j for j in _GO2}, last_action=[0.0] * 12)
        built = build_policy_obs(
            contract, _standing_obs(), joint_keys={j: j for j in _GO2}, last_action=[0.0] * 12,
            obs_terms={"contacts": [2.0] * 187},
        )  # fmt: skip
        assert built[48:] == [1.0] * 187  # the term's clip (-1, 1) still applies

    def test_an_incomplete_layout_is_refused(self) -> None:
        contract = complete_obs_layout(contract_from_io_descriptors(go2_io_descriptors()), num_actor_obs=235)
        with pytest.raises(DeployContractError, match="187 unaccounted"):
            build_policy_obs(contract, _standing_obs(), joint_keys={}, last_action=[0.0] * 12)


def _export_with_contract(tmp_path: Path, contract: dict[str, Any]) -> str:
    torch = pytest.importorskip("torch")
    from strands_robots.training.rl import rsl_rl
    from tests.training.test_rsl_rl_actor_export import write_rsl_rl_run

    model, _ = write_rsl_rl_run(tmp_path / "run")
    del torch
    return rsl_rl.convert_checkpoint(
        str(model), str(tmp_path / "out"), extra_meta={"task": "Isaac-X", "deploy_contract": contract}
    )


def _two_joint_contract() -> dict[str, Any]:
    from tests.training.test_isaaclab_deploy_contract import fake_io_descriptors

    env = {
        "observations": {
            "policy": {"joint_pos": {"func": "m:joint_pos_rel"}, "joint_vel": {"func": "m:joint_vel_rel"}}
        },
        "scene": {
            "robot": {
                "actuators": {
                    "all": {
                        "class_type": "m:IdealPDActuator",
                        "joint_names_expr": [".*"],
                        "stiffness": 40.0,
                        "damping": 1.0,
                        "effort_limit": 5.0,
                    }
                }
            }
        },  # fmt: skip
    }
    return attach_env_cfg(contract_from_io_descriptors(fake_io_descriptors()), env)


class TestThePolicyBuildsItsOwnObservation:
    def test_get_actions_reads_the_robot_state_and_feeds_back_its_last_action(self, tmp_path: Path) -> None:
        from strands_robots.policies import create_policy

        path = _export_with_contract(tmp_path, _two_joint_contract())
        policy = create_policy("rl", checkpoint_dir=path)
        policy.set_robot_state_keys(["slider_to_cart", "cart_to_pole"])
        state = {"slider_to_cart": 0.3, "cart_to_pole": -0.1, "slider_to_cart.vel": 0.0, "cart_to_pole.vel": 0.2}
        via_state = asyncio.run(policy.get_actions(state, ""))[0]
        reference = create_policy("rl", checkpoint_dir=path)
        reference.set_robot_state_keys(["slider_to_cart", "cart_to_pole"])
        # joint_pos_rel is relative to the default pose (0.1, -0.2), joint_vel_rel as read.
        via_vector = asyncio.run(reference.get_actions({"policy_obs": [0.2, 0.1, 0.0, 0.2]}, ""))[0]
        assert via_state == pytest.approx(via_vector)
        policy.reset()
        assert asyncio.run(policy.get_actions(state, ""))[0] == pytest.approx(via_state)

    def test_a_state_without_the_joints_is_refused_naming_them(self, tmp_path: Path) -> None:
        from strands_robots.policies import create_policy

        policy = create_policy("rl", checkpoint_dir=_export_with_contract(tmp_path, _two_joint_contract()))
        policy.set_robot_state_keys(["slider_to_cart", "cart_to_pole"])
        with pytest.raises(ValueError, match="observation cannot be built"):
            asyncio.run(policy.get_actions({"slider_to_cart": 0.0}, ""))


_MJCF = """
<mujoco><option timestep="0.005"/><worldbody><body name="link"><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="0.5"/></body></worldbody>
<actuator>{actuator}</actuator></mujoco>
"""


class _World:
    def __init__(self, model: Any) -> None:
        self._model = model
        self.robots: dict[str, Any] = {}
        self._backend_state: dict[str, Any] = {}


class TestTheActuatorPdRunsOnTorqueMotors:
    def test_a_torque_motor_is_driven_to_the_target_by_the_runs_pd(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        from strands_robots.policies.isaaclab_actuator_pd import ContractPDController

        model = mujoco.MjModel.from_xml_string(_MJCF.format(actuator='<motor name="hinge" joint="hinge"/>'))
        data = mujoco.MjData(model)
        sim = type("S", (), {"_world": _World(model)})()
        contract = {
            "actuators": {"hinge_joint": {"stiffness": 20.0, "damping": 0.5, "effort_limit": 10.0}},
            "control_dt": 0.02,
        }
        pd = ContractPDController.from_sim(sim, "r", contract, {"hinge_joint": "hinge"})
        assert pd is not None and pd._substeps == 4
        for _ in range(100):
            pd.apply({"hinge": 0.4}, model, data, "r")
        assert float(data.qpos[0]) == pytest.approx(0.4, abs=0.05)  # P-only: gravity leaves a small offset
        assert abs(float(data.ctrl[0])) <= 10.0

    def test_a_position_servo_is_left_alone(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        from strands_robots.policies.isaaclab_actuator_pd import ContractPDController

        model = mujoco.MjModel.from_xml_string(_MJCF.format(actuator='<position name="hinge" joint="hinge" kp="30"/>'))
        sim = type("S", (), {"_world": _World(model)})()
        contract = {"actuators": {"hinge_joint": {"stiffness": 20.0, "damping": 0.5}}, "control_dt": 0.02}
        assert ContractPDController.from_sim(sim, "r", contract, {"hinge_joint": "hinge"}) is None
