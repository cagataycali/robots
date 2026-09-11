# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A ``policy_object`` the rollout cannot drive is refused at the entry point.

``policy_object`` hands a rollout a pre-built policy, which is why it BYPASSES
the two checks that would otherwise judge it: provider resolution and the
provider's ``preflight`` hook. It is the third opaque policy parameter on this
surface, and the other two - ``policy_config`` and ``policy_kwargs`` - are
already held to their shape at the public entry point for exactly the reason
this one was not:

* ``run_policy(policy_object=42)`` raised a bare
  ``AttributeError: 'int' object has no attribute 'set_control_frequency'``,
  and ``eval_policy`` the same for ``set_robot_state_keys`` - a library
  internal, naming neither the parameter nor the surface. Passing the CLASS
  instead of an instance was worse still: attribute access on a class reaches
  unbound descriptors, so ``policy_object=MockPolicy`` surfaced as
  ``TypeError: 'property' object is not iterable`` from inside a tree walk.
* ``start_policy(policy_object=42)`` returned ``status="success"`` /
  "Policy started". That raise happened on the executor worker, whose result
  nothing reads, so the caller held a success for a rollout that applied no
  action - and ``list_policies_running`` then reported "No policies running.",
  the same reading a completed rollout gives.

The second case is the one ``start_policy``'s own contract rules out ("Every
request ``run_policy`` refuses is refused here too, before the submit"), so
the refusal is pinned as an EQUALITY between the blocking and async surfaces
rather than as a substring: the point is not that each says something, it is
that both say the same thing about the same request. Usable values are pinned
too - the guard must not cost a rollout that was going to work.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from strands_robots.policies import policy_object_error
from strands_robots.policies.mock import MockPolicy

pytest.importorskip("mujoco")

from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

# One wrong value per way a caller gets here: a bare scalar, the provider NAME
# (which belongs in policy_provider), a config dict (which belongs in
# policy_config), and the class instead of an instance.
UNDRIVABLE: list[tuple[str, Any]] = [
    ("scalar", 42),
    ("provider-name", "mock"),
    ("config-dict", {"provider": "mock"}),
    ("the-class", MockPolicy),
]


def _text(result: dict[str, Any]) -> str:
    """The human-readable half of an agent-tool envelope."""
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def sim(tmp_path):
    """One minimal arm, so a rollout has an actuator to drive."""
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    engine = Simulation(tool_name="policy_object_domain", mesh=False)
    try:
        engine.create_world()
        added = engine.add_robot(name="arm", urdf_path=str(xml_path))
        assert added["status"] == "success", added
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _wait_until_idle(engine: Simulation, timeout: float = 15.0) -> str:
    """Block until no rollout is in flight, then report what the surface says."""
    reported = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reported = _text(engine.list_policies_running())
        if "No policies running" in reported:
            return reported
        time.sleep(0.02)
    raise AssertionError(f"a rollout is still in flight after {timeout}s: {reported}")


# Every rollout facade that takes ``policy_object``. Held against the class
# signatures below, so a facade added later joins this table or fails.
FACADE_NAMES = ("eval_policy", "evaluate_benchmark", "run_policy", "start_policy")


def _facades(engine: Simulation) -> dict[str, Any]:
    """Every rollout facade that takes ``policy_object``, called the same way."""
    return {
        "run_policy": lambda value: engine.run_policy(
            robot_name="arm", policy_object=value, n_steps=3, control_frequency=30.0, fast_mode=True
        ),
        "start_policy": lambda value: engine.start_policy(
            robot_name="arm", policy_object=value, n_steps=3, control_frequency=30.0, fast_mode=True
        ),
        "eval_policy": lambda value: engine.eval_policy(
            robot_name="arm", policy_object=value, n_episodes=1, max_steps=3, control_frequency=30.0
        ),
        "evaluate_benchmark": lambda value: engine.evaluate_benchmark(
            "__no_such_benchmark__", robot_name="arm", policy_object=value, n_episodes=1
        ),
    }


class TestTheDomainIsWhatCanBeDriven:
    """The shared message builder: a Policy instance, or nothing."""

    @pytest.mark.parametrize("value", [None, MockPolicy()], ids=["omitted", "instance"])
    def test_a_drivable_value_is_accepted(self, value):
        assert policy_object_error(value) is None

    @pytest.mark.parametrize(("label", "value"), UNDRIVABLE, ids=[i for i, _ in UNDRIVABLE])
    def test_an_undrivable_value_names_the_parameter(self, label, value):
        message = policy_object_error(value)
        assert message is not None
        assert "policy_object" in message

    def test_a_policy_subclass_is_told_to_instantiate_itself(self):
        """The class is not an instance, and the remedy is to call it."""
        message = policy_object_error(MockPolicy)
        assert message is not None
        assert "MockPolicy()" in message, message


class TestEveryRolloutFacadeRefusesIt:
    """The guard sits on each facade, before anything is driven."""

    @pytest.mark.parametrize("facade", FACADE_NAMES)
    @pytest.mark.parametrize(("label", "value"), UNDRIVABLE, ids=[i for i, _ in UNDRIVABLE])
    def test_the_refusal_names_the_parameter_and_drives_nothing(self, sim, facade, label, value):
        applied: list[Any] = []
        real_send = sim.send_action
        sim.send_action = lambda *a, **k: (applied.append(a), real_send(*a, **k))[1]
        try:
            result = _facades(sim)[facade](value)
        finally:
            sim.send_action = real_send
        _wait_until_idle(sim)

        assert result["status"] == "error", result
        assert "policy_object" in _text(result), result
        # A refusal that let the rollout start would still be a rollout the
        # caller did not get: no action may reach the sim.
        assert applied == []


class TestBothSurfacesGiveOneVerdict:
    """``start_policy`` says exactly what ``run_policy`` says, not merely something."""

    @pytest.mark.parametrize(("label", "value"), UNDRIVABLE, ids=[i for i, _ in UNDRIVABLE])
    def test_the_async_surface_repeats_the_blocking_surface(self, sim, label, value):
        blocking = sim.run_policy(
            robot_name="arm", policy_object=value, n_steps=3, control_frequency=30.0, fast_mode=True
        )
        _wait_until_idle(sim)
        started = sim.start_policy(
            robot_name="arm", policy_object=value, n_steps=3, control_frequency=30.0, fast_mode=True
        )
        _wait_until_idle(sim)

        assert blocking["status"] == started["status"] == "error"
        # Same sentence, differing only in the surface that prefixes it.
        assert _text(blocking).removeprefix("run_policy: ") == _text(started).removeprefix("start_policy: ")

    def test_the_refusal_leaves_the_robot_claimable(self, sim):
        """A guard that returned after claiming the robot would wedge it busy."""
        refused = sim.start_policy(
            robot_name="arm", policy_object=42, n_steps=3, control_frequency=30.0, fast_mode=True
        )
        assert refused["status"] == "error", refused
        assert "No policies running" in _text(sim.list_policies_running())

        accepted = sim.start_policy(
            robot_name="arm", policy_object=MockPolicy(), n_steps=3, control_frequency=30.0, fast_mode=True
        )
        assert accepted["status"] == "success", accepted
        _wait_until_idle(sim)


class TestADrivablePolicyStillRuns:
    """The guard costs nothing a caller was going to get."""

    @pytest.mark.parametrize("facade", ["run_policy", "start_policy"])
    def test_an_instance_drives_the_arm_on_both_surfaces(self, sim, facade):
        applied: list[Any] = []
        real_send = sim.send_action
        sim.send_action = lambda *a, **k: (applied.append(a), real_send(*a, **k))[1]
        try:
            result = _facades(sim)[facade](MockPolicy())
            _wait_until_idle(sim)
        finally:
            sim.send_action = real_send

        assert result["status"] == "success", result
        assert applied, "a usable policy_object must still reach the actuators"


class TestTheGuardCoversEveryFacadeThatAcceptsIt:
    """Derived from the signatures, so a facade added later is covered or fails."""

    def test_every_public_method_taking_policy_object_is_exercised_here(self, sim):
        import inspect

        declared = {
            name
            for name, member in inspect.getmembers(Simulation, inspect.isfunction)
            if not name.startswith("_") and "policy_object" in inspect.signature(member).parameters
        }
        # run_multi_policy takes a MAPPING of policies rather than one object, so
        # it is a different parameter with a different domain.
        declared -= {"run_multi_policy"}
        assert declared == set(FACADE_NAMES) == set(_facades(sim)), declared

    def test_the_engine_guard_is_the_one_the_facades_call(self):
        """Non-vacuity: the shared helper exists and refuses through the envelope."""
        refusal = SimEngine._validate_policy_object(42, "run_policy")
        assert refusal is not None
        assert refusal["status"] == "error"
        assert "run_policy: policy_object" in _text(refusal)
        assert SimEngine._validate_policy_object(MockPolicy(), "run_policy") is None
