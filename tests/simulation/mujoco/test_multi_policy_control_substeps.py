# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""run_multi_policy must integrate a full control period per applied action.

The loop called ``mj_step`` exactly ONCE per control iteration and never derived
substeps from ``control_frequency``, unlike ``run_policy`` / ``eval_policy``,
which share ``PolicyRunner._control_substeps`` (documented as the "single source
of truth" for the derivation). At the 500Hz physics default (dt=0.002) and a
declared 50Hz control rate, each action integrated 2ms instead of 20ms - a 10x
under-integration.

``data.ctrl`` is held constant across ``mj_step``, so a position servo only
closes its error if given the whole period. Measured on the same scene, same
constant-target policy, same 100-step horizon at 50Hz:

    run_multi_policy  sim_time 0.2000s   joints {'2': 0.3383, '3': 0.396}
    PolicyRunner.run  sim_time 2.0000s   joints {'2': 0.29,   '3': 0.4553}

so multi-robot rollouts looked like the policy was a no-op, and every recorded
dataset timestamp was short by the same factor. This is the exact failure
``_control_substeps`` was introduced to prevent; the fix reached the
single-robot runner but never the multi-robot loop.

Gated on mujoco: every assertion steps real physics.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_JOINTS = [str(i) for i in range(1, 7)]
_TARGET = 0.6


class _ConstantTarget(Policy):
    """Commands the same joint target forever, so tracking is measurable."""

    @property
    def provider_name(self) -> str:
        return "constant"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(_JOINTS, _TARGET)]


def _sim_with_arm(name: str = "alice") -> MuJoCoSimEngine:
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot(name, data_config="so101")["status"] == "success"
    return sim


def _joints(sim, robot: str) -> dict[str, float]:
    obs = sim.get_observation(robot, skip_images=True)
    return {k: float(v) for k, v in obs.items() if not k.endswith(".vel") and isinstance(v, float)}


class TestSimTimeMatchesTheDeclaredControlRate:
    def test_sim_time_advances_one_control_period_per_step(self):
        """Regression: this advanced one physics dt (10x short) per step."""
        sim = _sim_with_arm()
        try:
            steps, hz = 100, 50.0
            assert (
                sim.run_multi_policy(policies={"alice": _ConstantTarget()}, n_steps=steps, control_frequency=hz)[
                    "status"
                ]
                == "success"
            )

            assert sim._world.sim_time == pytest.approx(steps / hz, rel=1e-6)
        finally:
            sim.destroy()

    def test_duration_contract_is_honoured(self):
        """The docstring calls duration "episode length in seconds"; make it true."""
        sim = _sim_with_arm()
        try:
            sim.run_multi_policy(policies={"alice": _ConstantTarget()}, duration=1.0, control_frequency=50.0)

            assert sim._world.sim_time == pytest.approx(1.0, rel=0.02)
        finally:
            sim.destroy()

    @pytest.mark.parametrize("hz", [25.0, 50.0, 100.0])
    def test_holds_across_control_rates(self, hz):
        sim = _sim_with_arm()
        try:
            steps = 40
            sim.run_multi_policy(policies={"alice": _ConstantTarget()}, n_steps=steps, control_frequency=hz)

            assert sim._world.sim_time == pytest.approx(steps / hz, rel=1e-6)
        finally:
            sim.destroy()


class TestParityWithTheSingleRobotRunner:
    def test_same_scene_and_target_reach_the_same_joint_state(self):
        """The two drivers must not disagree about the physics they ran."""
        multi = _sim_with_arm()
        single = _sim_with_arm()
        try:
            kwargs = dict(n_steps=100, control_frequency=50.0)
            multi.run_multi_policy(policies={"alice": _ConstantTarget()}, **kwargs)
            single.run_policy(policy_object=_ConstantTarget(), robot_name="alice", **kwargs)

            multi_joints = _joints(multi, "alice")
            single_joints = _joints(single, "alice")
            assert set(multi_joints) == set(single_joints)
            for joint, value in single_joints.items():
                assert multi_joints[joint] == pytest.approx(value, abs=1e-3), (
                    f"{joint}: multi={multi_joints[joint]:.4f} single={value:.4f}"
                )
        finally:
            multi.destroy()
            single.destroy()

    def test_servo_actually_converges_on_the_target(self):
        """Pre-fix several joints sat near zero or fell under gravity."""
        sim = _sim_with_arm()
        try:
            sim.run_multi_policy(policies={"alice": _ConstantTarget()}, n_steps=150, control_frequency=50.0)

            reached = _joints(sim, "alice")
            # The distal joints track a 0.6 rad target closely once the full
            # control period is integrated.
            assert reached["6"] == pytest.approx(_TARGET, abs=0.05), reached
            assert reached["5"] == pytest.approx(_TARGET, abs=0.05), reached
        finally:
            sim.destroy()


class TestControlSubstepsOverride:
    def test_explicit_override_is_honoured(self):
        sim = _sim_with_arm()
        try:
            steps, substeps = 10, 3
            sim.run_multi_policy(
                policies={"alice": _ConstantTarget()},
                n_steps=steps,
                control_frequency=50.0,
                control_substeps=substeps,
            )

            dt = sim.physics_timestep()
            assert sim._world.sim_time == pytest.approx(steps * substeps * dt, rel=1e-6)
        finally:
            sim.destroy()

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_invalid_override_raises_rather_than_being_clamped(self, bad):
        """Clamping would silently reinstate the under-integration."""
        sim = _sim_with_arm()
        try:
            with pytest.raises(ValueError, match="control_substeps"):
                sim.run_multi_policy(
                    policies={"alice": _ConstantTarget()},
                    n_steps=2,
                    control_frequency=50.0,
                    control_substeps=bad,
                )
        finally:
            sim.destroy()


class TestStepAccounting:
    def test_step_count_reflects_physics_steps_taken(self):
        """step_count counts PHYSICS steps, so it must include the substeps."""
        sim = _sim_with_arm()
        try:
            steps, hz = 20, 50.0
            before = sim._world.step_count
            sim.run_multi_policy(policies={"alice": _ConstantTarget()}, n_steps=steps, control_frequency=hz)

            expected_substeps = round((1.0 / hz) / sim.physics_timestep())
            assert sim._world.step_count - before == steps * expected_substeps
        finally:
            sim.destroy()

    def test_per_robot_policy_step_count_is_unchanged(self):
        """Policy steps are CONTROL steps; the fix must not inflate them."""
        sim = _sim_with_arm()
        try:
            result = sim.run_multi_policy(policies={"alice": _ConstantTarget()}, n_steps=10, control_frequency=50.0)

            assert result["status"] == "success"
            assert sim._world.robots["alice"].policy_steps == 10
        finally:
            sim.destroy()
