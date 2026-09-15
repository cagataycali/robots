"""A rollout that never commanded the robot is refused, not reported complete.

MODULE ``strands_robots.simulation.policy_runner``.

:meth:`~strands_robots.simulation.policy_runner.PolicyRunner.run` already
refuses the neighbouring shape: when every step emits keys and NONE of them
resolve to an actuator, the rollout is an error naming the unresolved keys. The
mirror - every step emits NO key at all - reached the same physical outcome (the
robot is never commanded) and was reported ``status="success"``, because a
policy that names nothing produces no unresolved key to count.

Both routes to it are reachable from shipped code. A robot whose model declares
no actuator binds a policy to an empty action-key list, so every action it emits
is empty; and a policy whose decode yields rows with no joint value emits the
same empty action on a fully actuated robot
(:meth:`CuroboPolicy._next_chunk` does exactly that per waypoint when the
planner's rows carry no joint position).

Neither was visible in the payload. ``action_resolution_rate`` - the field
:func:`~strands_robots.simulation.policy_runner.action_commands_robot` names as
the way this surface separates "commanded everything" from "commanded nothing" -
is keyed on the robot's actuators, so it is an EMPTY map when there are none,
which reads exactly like a robot with no resolution problem. Beside it
``action_errors=0`` and ``partial_action_failure_rate=0.0`` positively assert
health.

These tests pin the aggregate verdict (the per-step tolerance is unchanged: a
single empty action is legitimate policy behaviour) and the count that backs it.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots
from strands_robots.policies import MockPolicy
from strands_robots.policies.base import Policy

# Two joints, two position actuators. ``angle="radian"`` is required - MJCF
# defaults to degrees, which would make the ranges ~2 degrees.
ACTUATED_XML = """
<mujoco model="commanded_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="link" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" ctrlrange="-2 2"/>
    <position name="elbow_act" joint="elbow" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

# The same two joints with the ``<actuator>`` block removed: a model that is
# fully articulated and wholly uncommandable. This is not a contrived shape -
# the shipped ``talos`` (45 joints) and ``asimov_v0`` (15 joints) descriptions
# compile to exactly it, so ``robot_action_keys`` reports zero keys for both.
UNACTUATED_XML = ACTUATED_XML.replace(
    """  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" ctrlrange="-2 2"/>
    <position name="elbow_act" joint="elbow" kp="30" ctrlrange="-2 2"/>
  </actuator>
""",
    "",
)


class NamesNoKey(Policy):
    """A real policy whose every action commands nothing.

    Reached two ways in shipped code: bound to a robot with no actuator (the
    key list is empty, so the comprehension a provider builds is empty too), or
    a decode whose rows carry no joint value. Counts its calls so a test can
    tell "the policy was never asked" from "it was asked and named nothing".
    """

    provider_name = "names_no_key"

    def __init__(self) -> None:
        self.calls = 0

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return [{}]


class DrivesOneJoint(Policy):
    """A real policy that commands exactly one of the robot's two actuators.

    The control for the documented partial posture: under-actuation is a
    ``success`` carrying ``partial_action_failure_rate``, and must stay one.
    """

    provider_name = "drives_one_joint"

    def __init__(self, key: str) -> None:
        self.key = key

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{self.key: 0.1}]


def payload_of(result: dict[str, Any]) -> dict[str, Any]:
    """The first ``{"json": {...}}`` block, found by scanning - never by index."""
    return next((b["json"] for b in result.get("content", []) if isinstance(b.get("json"), dict)), {})


def text_of(result: dict[str, Any]) -> str:
    return " ".join(b["text"] for b in result.get("content", []) if isinstance(b.get("text"), str))


def _engine(tmp_path, xml: str, name: str):
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    engine = strands_robots.Simulation(backend="mujoco", tool_name="commanded_guard", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(path))
    return engine


@pytest.fixture
def actuated(tmp_path):
    engine = _engine(tmp_path, ACTUATED_XML, "actuated")
    yield engine
    engine.cleanup()


@pytest.fixture
def unactuated(tmp_path):
    engine = _engine(tmp_path, UNACTUATED_XML, "unactuated")
    yield engine
    engine.cleanup()


def _run(engine, policy, steps: int = 4) -> dict[str, Any]:
    policy.set_robot_state_keys(engine.robot_action_keys("arm"))
    return engine.run_policy(
        robot_name="arm", policy_object=policy, n_steps=steps, control_frequency=50.0, action_horizon=1
    )


class TestThePremisesTheGuardRestsOn:
    """Each route really does reach a rollout that commands nothing."""

    def test_a_model_with_no_actuator_block_reports_no_action_key(self, unactuated):
        assert unactuated.robot_action_keys("arm") == []
        # Still fully articulated: the joints exist, only the drives are absent,
        # so this is under-actuation and not an empty scene.
        assert unactuated.robot_joint_names("arm") == ["shoulder", "elbow"]

    def test_the_actuated_control_reports_both_keys(self, actuated):
        assert actuated.robot_action_keys("arm") == ["shoulder_act", "elbow_act"]

    def test_the_policy_is_asked_and_names_nothing(self, unactuated):
        policy = NamesNoKey()
        _run(unactuated, policy, steps=4)
        assert policy.calls == 4, "the policy must be exercised, or the guard proves nothing"


class TestARolloutThatCommandedNothingIsAnError:
    """The aggregate verdict, by both routes to it."""

    def test_a_robot_with_no_actuator_is_refused(self, unactuated):
        result = _run(unactuated, NamesNoKey())
        assert result["status"] == "error", result

    def test_a_policy_naming_no_key_is_refused_on_an_actuated_robot(self, actuated):
        result = _run(actuated, NamesNoKey())
        assert result["status"] == "error", result

    @pytest.mark.parametrize("fixture_name", ["unactuated", "actuated"])
    def test_the_refusal_names_the_robot_and_the_remedy(self, request, fixture_name):
        engine = request.getfixturevalue(fixture_name)
        text = text_of(_run(engine, NamesNoKey()))
        assert "arm" in text
        assert "commanded no actuator" in text
        # The two routes are distinguished by the robot's own key count, which is
        # what tells "this model has no drives" from "the policy decoded nothing".
        assert "robot_action_keys" in text

    def test_an_error_result_reports_stopped_reason_error(self, unactuated):
        # Mirrors the sibling total-unresolved refusal: a full step budget that
        # ended in an error is not a retryable "budget" completion.
        assert payload_of(_run(unactuated, NamesNoKey()))["stopped_reason"] == "error"


class TestTheCountThatBacksTheVerdict:
    """``actions_applied`` counts actions that commanded a key, not send calls."""

    def test_a_rollout_that_commanded_nothing_reports_zero(self, unactuated):
        # Not the number of ``send_action`` calls: an action commanding nothing
        # reaches the backend like any other and would count four.
        assert payload_of(_run(unactuated, NamesNoKey()))["actions_applied"] == 0

    def test_a_healthy_rollout_counts_every_commanded_action(self, actuated):
        policy = MockPolicy()
        result = _run(actuated, policy, steps=4)
        assert result["status"] == "success", result
        assert payload_of(result)["actions_applied"] == 4

    def test_the_payload_carries_the_count_on_a_healthy_rollout(self, actuated):
        assert "actions_applied" in payload_of(_run(actuated, MockPolicy()))


class TestThePerStepToleranceIsUnchanged:
    """Only the aggregate is refused; partial behaviour keeps its posture."""

    def test_a_policy_driving_one_of_two_actuators_is_still_a_success(self, actuated):
        result = _run(actuated, DrivesOneJoint("shoulder_act"), steps=4)
        assert result["status"] == "success", result
        payload = payload_of(result)
        assert payload["actions_applied"] == 4
        # The documented posture: under-actuation is visible in the rate, not in
        # the status.
        assert payload["partial_action_failure_rate"] == pytest.approx(0.5)

    def test_a_single_empty_action_among_commanded_ones_is_not_refused(self, actuated):
        class OneStall(Policy):
            provider_name = "one_stall"

            def __init__(self) -> None:
                self.n = 0

            def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
                self.keys = list(robot_state_keys)

            async def get_actions(
                self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
            ) -> list[dict[str, Any]]:
                self.n += 1
                return [{} if self.n == 1 else {k: 0.1 for k in self.keys}]

        result = _run(actuated, OneStall(), steps=4)
        assert result["status"] == "success", result
        assert payload_of(result)["actions_applied"] == 3
