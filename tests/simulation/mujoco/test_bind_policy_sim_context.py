"""Behavior tests for ``Simulation.bind_policy_sim_context``.

``bind_policy_sim_context`` is the seam that hands the compiled ``MjModel`` and
a robot's joint namespace to policies that opt in via a ``set_sim_context``
method (e.g. ``VeraPolicy`` auto-discovering its end-effector frame for
zero-config IK). The contract this method pins:

* It is a strict no-op for policies that do not expose a callable
  ``set_sim_context`` - ordinary joint-position policies are unaffected.
* It guards every precondition (no world, uncompiled model, unknown robot)
  and returns silently rather than raising, so a sim that is mid-build never
  crashes a rollout through this path.
* When all preconditions hold it forwards the live ``MjModel`` and the
  robot's namespace string to the policy exactly once.
* A binding error raised inside the policy's ``set_sim_context`` is swallowed
  (logged at debug) - a policy's IK auto-config failure must degrade to a
  joint-space rollout, never abort it.

These branches were previously unexercised on the MuJoCo backend; a refactor
that turned the fail-soft swallow into a propagated exception would silently
start crashing eef-delta rollouts.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation


class _RecordingContextPolicy:
    """Minimal policy stand-in that records ``set_sim_context`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def set_sim_context(self, mj_model: Any, namespace: Any) -> None:
        self.calls.append((mj_model, namespace))


class _RaisingContextPolicy:
    """Policy whose IK auto-config raises - must be swallowed by the binder."""

    def __init__(self) -> None:
        self.called = False

    def set_sim_context(self, mj_model: Any, namespace: Any) -> None:
        self.called = True
        raise RuntimeError("ee-frame discovery blew up")


class _NoContextPolicy:
    """Ordinary joint-position policy with no ``set_sim_context`` attribute."""


class _NonCallableContextPolicy:
    """Edge case: ``set_sim_context`` exists but is not callable."""

    set_sim_context = "not a method"


@pytest.fixture
def sim_with_robot():
    s = Simulation()
    s.create_world()
    s.add_robot("so100")
    try:
        yield s
    finally:
        s.cleanup()


class TestBindPolicySimContext:
    """The opt-in MjModel/namespace handoff contract."""

    def test_forwards_model_and_namespace_when_preconditions_hold(self, sim_with_robot):
        """A policy that opts in receives the live model + robot namespace once."""
        policy = _RecordingContextPolicy()

        sim_with_robot.bind_policy_sim_context(policy, "so100")

        assert len(policy.calls) == 1
        model_arg, namespace_arg = policy.calls[0]
        # The forwarded model is the world's compiled MjModel, not a copy.
        assert model_arg is sim_with_robot._world._model
        # The namespace is the robot's namespace string (always a str).
        assert namespace_arg == sim_with_robot._world.robots["so100"].namespace
        assert isinstance(namespace_arg, str)

    def test_noop_for_policy_without_set_sim_context(self, sim_with_robot):
        """Ordinary policies (no set_sim_context) are untouched and never error."""
        policy = _NoContextPolicy()
        # Must not raise and must not mutate anything observable.
        sim_with_robot.bind_policy_sim_context(policy, "so100")
        assert not hasattr(policy, "calls")

    def test_noop_when_set_sim_context_not_callable(self, sim_with_robot):
        """A non-callable set_sim_context attribute is ignored, not invoked."""
        policy = _NonCallableContextPolicy()
        sim_with_robot.bind_policy_sim_context(policy, "so100")
        # Attribute is unchanged - it was never treated as a method.
        assert policy.set_sim_context == "not a method"

    def test_noop_when_no_world(self):
        """Before create_world there is no model to bind - silent no-op."""
        s = Simulation()
        policy = _RecordingContextPolicy()
        # No world built yet; must guard and return without invoking the policy.
        s.bind_policy_sim_context(policy, "so100")
        assert policy.calls == []

    def test_noop_for_unknown_robot(self, sim_with_robot):
        """An unknown robot name short-circuits before invoking the policy."""
        policy = _RecordingContextPolicy()
        sim_with_robot.bind_policy_sim_context(policy, "ghost_arm")
        assert policy.calls == []

    def test_binding_error_is_swallowed(self, sim_with_robot, caplog):
        """A set_sim_context that raises must not propagate - rollout-safe."""
        policy = _RaisingContextPolicy()
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.mujoco.simulation"):
            # Must not raise despite the policy throwing inside set_sim_context.
            sim_with_robot.bind_policy_sim_context(policy, "so100")
        assert policy.called is True
        assert any("bind_policy_sim_context" in rec.message for rec in caplog.records)
