"""A ``policy_object`` the caller bound to some of the robot's keys keeps them through a rollout.

``run_policy`` / ``eval_policy`` / ``evaluate_benchmark`` called
``policy.set_robot_state_keys(robot_action_keys)`` on a pre-built policy too, so a
pi0.5-DROID policy set to the panda's 7 arm joints + one finger (8 of its 9 keys)
was re-bound to all 9 on every rollout and could not be driven as trained. A
caller's own keys are kept when every one is this robot's; placeholders and
foreign keys are still replaced.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.simulation import create_simulation  # noqa: E402


@pytest.fixture
def sim():
    engine = create_simulation("mujoco", mesh=False)
    engine.create_world()
    engine.add_robot("so101")
    yield engine
    engine.cleanup()


def _robot(sim) -> str:
    return sim.list_robots()[0]


def test_a_subset_the_caller_chose_is_kept(sim) -> None:
    keys = sim.robot_action_keys(_robot(sim))
    policy = create_policy("mock")
    policy.set_robot_state_keys(keys[:5])
    result = sim.run_policy(robot_name=_robot(sim), policy_object=policy, n_steps=2, control_frequency=50.0)
    assert result["status"] == "success", result
    assert list(policy.robot_state_keys) == keys[:5]


@pytest.mark.parametrize("given", [["joint_0", "joint_1"], ["not_a_joint"], []])
def test_placeholders_foreign_or_missing_keys_are_rebound(sim, given: list[str]) -> None:
    keys = sim.robot_action_keys(_robot(sim))
    policy = create_policy("mock")
    if given:
        policy.set_robot_state_keys(given)
    sim.run_policy(robot_name=_robot(sim), policy_object=policy, n_steps=1, control_frequency=50.0)
    assert list(policy.robot_state_keys) == keys


def test_a_policy_built_by_the_call_gets_the_robots_keys(sim) -> None:
    keys = sim.robot_action_keys(_robot(sim))
    seen: dict[str, list[str]] = {}
    original = sim._bind_policy_state_keys

    def spy(policy, robot_name, *, prebuilt):
        original(policy, robot_name, prebuilt=prebuilt)
        seen["keys"] = list(policy.robot_state_keys)

    sim._bind_policy_state_keys = spy
    sim.run_policy(robot_name=_robot(sim), policy_provider="mock", n_steps=1, control_frequency=50.0)
    assert seen["keys"] == keys
