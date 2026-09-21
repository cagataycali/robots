"""Per-actuator resolution stats must credit what ``send_action`` resolves.

``Simulation.send_action`` accepts an ordered numeric vector (list / tuple /
1-D array) and binds it positionally to ``robot_action_keys(robot_name)``, so a
policy may emit a raw action vector per tick instead of a ``{joint: value}``
dict, and it resolves a dict key that names either an actuator or a driven JOINT
(looking that joint's driving actuator up, tendon grippers included).
``PolicyRunner`` tracks per-actuator resolution (issue #165): the fraction of
steps each actuator was actually driven, surfaced as ``action_resolution_rate``
and aggregated into ``partial_action_failure_rate``.

Both are credited against the robot's actuator roster, and anything that roster
cannot account for must be reported as UNKNOWN rather than as a measured miss -
``action_resolution_rate`` all 0.0 with ``partial_action_failure_rate`` 1.0 is the
signature of a rollout that never moved, and reporting it for one that drove the
robot every tick misdirects the caller the field exists for. Two spellings reach
that roster indirectly and are pinned here: a numeric vector, which names no key
at all and binds positionally to every actuator (so every actuator is credited),
and a dict keyed by driven joint names, whose resolved actuator the backend does
not report per key (so the step is left out of the denominator).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner


class _VectorActionPolicy(Policy):
    """Minimal policy that emits one positional action vector per tick."""

    def __init__(self, n_actuators: int) -> None:
        self._n = n_actuators

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[Any]:
        # A single-action chunk whose element is a numeric vector, not a dict.
        return [[0.0] * self._n]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "vector_test"


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="vector_action_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    for block in result.get("content", []):
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError("result has no json payload block")


class TestVectorActionResolutionStats:
    def test_vector_action_credits_every_actuator(self, sim_with_robot):
        """Every actuator resolves each step -> resolution 1.0, no under-actuation."""
        actuators = sim_with_robot.robot_action_keys("alice")
        policy = _VectorActionPolicy(len(actuators))
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        result = PolicyRunner(sim_with_robot).run(
            "alice",
            policy,
            duration=0.2,
            control_frequency=50,
            fast_mode=True,
        )

        assert result["status"] == "success"
        payload = _payload(result)
        resolution = payload["action_resolution_rate"]
        # A vector binds positionally to every actuator, so each is driven every step.
        assert set(resolution) == set(actuators)
        assert all(rate == 1.0 for rate in resolution.values()), resolution
        # Aggregate: nothing under-driven, no step-level send_action error.
        assert payload["partial_action_failure_rate"] == 0.0
        assert payload["action_errors"] == 0


class _DictActionPolicy(Policy):
    """Minimal policy that emits one ``{key: value}`` action dict per tick."""

    def __init__(self, keys: list[str]) -> None:
        self._action_keys = list(keys)

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[Any]:
        return [dict.fromkeys(self._action_keys, 0.0)]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "dict_test"


#: Panda joint names that each have a driving actuator - the arm's seven position
#: servos plus the finger the tendon gripper drives. ``finger_joint2`` is the mimic
#: follower and has no actuator of its own, so naming it would be a real
#: unresolved key and would make this a test about refusals instead.
_PANDA_DRIVEN_JOINTS = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]


@pytest.fixture
def panda_sim():
    """A robot whose actuator names differ from its joint names (``actuator1..8``)."""
    s = Simulation(tool_name="dict_action_test", mesh=False)
    s.create_world()
    s.add_robot(name="arm", data_config="panda")
    yield s
    s.cleanup()


class TestDictActionResolutionStats:
    """A dict of driven joint names drove the robot, so it is not under-actuation."""

    def test_actuator_keys_are_credited_to_each_actuator(self, panda_sim):
        """The direct spelling: every actuator named every step -> 1.0 each."""
        actuators = panda_sim.robot_action_keys("arm")
        result = PolicyRunner(panda_sim).run(
            "arm", _DictActionPolicy(actuators), duration=0.2, control_frequency=50, fast_mode=True
        )
        assert result["status"] == "success", result
        payload = _payload(result)
        assert set(payload["action_resolution_rate"]) == set(actuators)
        assert all(rate == 1.0 for rate in payload["action_resolution_rate"].values()), payload
        assert payload["partial_action_failure_rate"] == 0.0

    def test_driven_joint_keys_are_unknown_not_a_miss(self, panda_sim):
        """The indirect spelling: no actuator is named, so none is scored a miss."""
        actuators = set(panda_sim.robot_action_keys("arm"))
        # The precondition that makes this a test: not one key spells an actuator.
        assert actuators.isdisjoint(_PANDA_DRIVEN_JOINTS), actuators
        result = PolicyRunner(panda_sim).run(
            "arm", _DictActionPolicy(_PANDA_DRIVEN_JOINTS), duration=0.2, control_frequency=50, fast_mode=True
        )
        # The backend resolved every key, so the rollout is operational...
        assert result["status"] == "success", result
        payload = _payload(result)
        assert payload["action_errors"] == 0, payload
        assert payload["actions_applied"] > 0, payload
        # ...and the per-actuator aggregate must say "unknown", not "never driven".
        assert payload["action_resolution_rate"] == {}, payload
        assert payload["partial_action_failure_rate"] == 0.0, payload
