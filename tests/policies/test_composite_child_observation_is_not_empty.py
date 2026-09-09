"""A CompositePolicy child is never queried with an observation of no keys.

``lower_obs_keys`` / ``upper_obs_keys`` select the observation each child is
queried with. The selection is by name, and the names belong to whatever
produced the observation: a MuJoCo world names them after the robot's own joints
(``"1"``, ``"1.vel"``), while a LeRobot dataset spells the same reading
``"observation.state"``. A subset written in the wrong namespace therefore
shares no key at all, and the child was handed ``{}`` on every tick::

    c = CompositePolicy(locomotion, arms, lower_obs_keys=["observation.state"])
    # world observation: {"1": ..., "1.vel": ..., "2": ...}
    # -> locomotion queried with {} on every tick, rollout reports success

Measured on a 60-tick ``run_policy`` rollout of ``so101`` with a state-reading
lower child: 60 of 60 ticks blind, the joints it owns left 0.418 rad from where
the same composite put them with the subset spelled correctly, and
``status="success"`` both times.

The same class already refuses the mirror mistake on the ACTION side - a
``lower_joints`` group sharing no name with what the child emits raises, naming
the group and the emitted names - and ``RobotRLEnv`` refuses a configured
``actor_obs_keys`` the engine does not produce ("Validate obs keys up front so a
typo fails loudly here, not mid-rollout"). These pin the observation side onto
the same rule, plus the two boundaries it deliberately keeps: a partly satisfied
subset is still honored, and an observation with no keys at all is left alone.
"""

import asyncio

import pytest

from strands_robots.policies import CompositePolicy
from tests.policies.test_composite import StubPolicy, _run

WORLD_OBS = {"1": 0.0, "1.vel": 0.0, "2": 0.1, "2.vel": 0.0}
LOWER_TICK = {"1": 1.0, "2": 1.0}
UPPER_TICK = {"3": 2.0, "4": 2.0}

#: The seat under test, its ``*_obs_keys`` kwarg, and the child it starves.
SEATS = [("lower", "lower_obs_keys"), ("upper", "upper_obs_keys")]


def _composite(**kwargs):
    """A composite over disjoint groups whose children record their observation."""
    lower = StubPolicy([dict(LOWER_TICK)], name="locomotion")
    upper = StubPolicy([dict(UPPER_TICK)], name="manipulation")
    return CompositePolicy(lower, upper, lower_joints=["1", "2"], upper_joints=["3", "4"], **kwargs), lower, upper


class TestASubsetThatSelectsNothingIsRefused:
    """A key subset sharing no name with the observation refuses the tick."""

    @pytest.mark.parametrize(("role", "kwarg"), SEATS)
    def test_either_seat_refuses_and_names_what_to_fix(self, role, kwarg):
        c, lower, upper = _composite(**{kwarg: ["observation.state"]})
        child = lower if role == "lower" else upper

        with pytest.raises(ValueError) as exc:
            _run(c, obs=dict(WORLD_OBS))

        message = str(exc.value)
        assert child.provider_name in message, message
        assert kwarg in message, message
        assert "observation.state" in message, message
        # The observation's own keys, so the caller can read off the namespace
        # they should have written the subset in.
        assert "'1.vel'" in message, message

    @pytest.mark.parametrize(("role", "kwarg"), SEATS)
    def test_neither_child_is_queried(self, role, kwarg):
        """The refusal precedes the inference, so no child runs on a bad config."""
        c, lower, upper = _composite(**{kwarg: ["observation.state"]})

        with pytest.raises(ValueError):
            _run(c, obs=dict(WORLD_OBS))

        assert lower.last_obs is None, "the lower child was queried anyway"
        assert upper.last_obs is None, "the upper child was queried anyway"


class TestTheBoundariesTheRuleKeeps:
    """What stays honored: a partial subset, and an observation with no keys."""

    def test_a_partly_satisfied_subset_forwards_the_keys_that_are_there(self):
        """A child that got some of what it asked for still has a reading to act on."""
        c, lower, _ = _composite(lower_obs_keys=["1", "observation.state"])

        merged = _run(c, obs=dict(WORLD_OBS))

        assert lower.last_obs == {"1": 0.0}
        assert merged[0] == {"1": 1.0, "2": 1.0, "3": 2.0, "4": 2.0}

    def test_an_observation_with_no_keys_is_left_alone(self):
        """Nothing was produced, so the subset missed nothing - as on the action side."""
        c, lower, upper = _composite(lower_obs_keys=["observation.state"])

        merged = _run(c, obs={})

        assert lower.last_obs == {}
        assert upper.last_obs == {}
        assert merged[0] == {"1": 1.0, "2": 1.0, "3": 2.0, "4": 2.0}


class TestTheRolloutReportsIt:
    """``run_policy`` surfaces the refusal instead of a successful rollout.

    The composite is built over the robot's own actuator names so the rollout is
    one a correctly configured caller would get - only the subset's namespace is
    wrong. Before the rule, this rollout reported ``status="success"`` with the
    lower child queried blind on every tick.
    """

    def test_run_policy_returns_an_error_envelope_naming_the_subset(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="composite_blind_child", mesh=False)
        try:
            sim.create_world()
            sim.add_robot(name="so100", data_config="so100")
            keys = sim.robot_action_keys("so100")
            half = len(keys) // 2
            lower = StubPolicy([{k: 0.1 for k in keys[:half]}], name="locomotion")
            upper = StubPolicy([{k: 0.1 for k in keys[half:]}], name="manipulation")
            c = CompositePolicy(
                lower,
                upper,
                lower_joints=keys[:half],
                upper_joints=keys[half:],
                lower_obs_keys=["observation.state"],
            )

            result = sim.run_policy(robot_name="so100", policy_object=c, n_steps=3, control_frequency=25.0)

            text = "".join(part.get("text", "") for part in result["content"])
            assert result["status"] == "error", text
            assert "lower_obs_keys" in text, text
            # The engine's own key namespace is what the caller needs to see.
            assert keys[0] in text, text
            assert lower.last_obs is None, "the child ran on an empty observation"
        finally:
            sim.cleanup()


def test_the_full_observation_is_still_forwarded_when_no_subset_is_configured():
    """The default path is untouched: no subset, no rule to apply."""
    c, lower, upper = _composite()

    asyncio.run(c.get_actions(dict(WORLD_OBS), ""))

    assert lower.last_obs == WORLD_OBS
    assert upper.last_obs == WORLD_OBS
