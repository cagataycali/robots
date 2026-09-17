"""A ``stop_policy`` that stopped nothing names the rollouts it did not stop.

An empty ``robot_name`` resolves to the sole rollout and is refused naming what
is running (#3788), but a *named* stop that finds nothing was answered "Was not
running on '<robot>'" whatever else was in flight: the same sentence, and the
same ``status="success"``, whether the world was idle or another arm was
mid-rollout. ``docs/simulation/rollouts.md`` reserves that reading for "the
genuinely idempotent case, where nothing is in flight at all", so an agent that
aimed its stop at the wrong arm was told the stop was a no-op and nothing about
the motion still running.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

STOP_TIMEOUT = 5.0


def _text(result: dict[str, Any]) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _await_in_flight(sim: MuJoCoSimEngine, *names: str) -> None:
    deadline = time.monotonic() + STOP_TIMEOUT
    while not set(names) <= set(sim._rollouts_in_flight()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert set(names) <= set(sim._rollouts_in_flight()), f"{names} never reached the in-flight population"


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="stop_verdict", mesh=False)
    engine.create_world()
    for name in ("alpha", "beta", "gamma"):
        assert engine.add_robot(name=name, data_config="so101")["status"] == "success"
    yield engine
    engine.cleanup()


def _start(sim: MuJoCoSimEngine, name: str) -> None:
    result = sim.start_policy(name, policy_provider="mock", duration=30.0, control_frequency=30.0)
    assert result["status"] == "success", _text(result)


def test_a_stop_that_halted_nothing_names_the_rollout_it_did_not_halt(sim) -> None:
    """The harm: a no-op stop read as an idle fleet while the other arm drove."""
    _start(sim, "beta")
    _await_in_flight(sim, "beta")
    try:
        result = sim.stop_policy("alpha")
        assert result["status"] == "success"
        assert _json(result)["was_running"] is False
        assert _text(result) == (
            "Was not running on 'alpha'. A policy is running on 'beta'. Stop it first: action='stop_policy'."
        )
        assert "beta" in sim._rollouts_in_flight(), "the reply must name a rollout the stop really left running"
    finally:
        sim.stop_policy("beta")


def test_an_idle_world_keeps_the_bare_verdict(sim) -> None:
    """The form the docs reserve for "nothing is in flight at all" stays exactly that."""
    result = sim.stop_policy("alpha")
    assert result["status"] == "success"
    assert _text(result) == "Was not running on 'alpha'"


def test_several_rollouts_are_all_named_with_a_remedy_the_tool_accepts(sim) -> None:
    """The remedy is asked of the resolver: the bare form is refused with two in flight."""
    for name in ("beta", "gamma"):
        _start(sim, name)
    _await_in_flight(sim, "beta", "gamma")
    try:
        assert _text(sim.stop_policy("alpha")) == (
            "Was not running on 'alpha'. A policy is running on 'beta', 'gamma'. "
            "Stop them first: action='stop_policy' with robot_name='beta', then robot_name='gamma'."
        )
        assert sim.stop_policy("")["status"] == "error", "the bare form must not resolve here"
    finally:
        for name in ("beta", "gamma"):
            sim.stop_policy(name)


def test_a_stop_that_did_halt_a_rollout_reports_only_that(sim) -> None:
    """Naming the others belongs to the no-op verdict; a real stop is unchanged."""
    for name in ("beta", "gamma"):
        _start(sim, name)
    _await_in_flight(sim, "beta", "gamma")
    try:
        assert _text(sim.stop_policy("beta")) == "Stopped on 'beta'"
    finally:
        sim.stop_policy("gamma")


def test_a_backend_with_no_rollout_registry_keeps_the_bare_verdict() -> None:
    """Called directly: such a backend refuses in ``stop_policy`` before the verdict.

    ``_MinimalEngine`` keeps no durable per-robot claim either, so its own
    ``stop_policy`` refuses earlier and cannot reach this sentence - but the
    ``None`` population is the case a third-party backend with a durable claim
    and no registry lands on, and an affirmative claim about robots it cannot
    enumerate is the thing every surface here declines to make.
    """
    from tests.simulation.test_stop_policy_base_contract import _MinimalEngine

    engine = _MinimalEngine()
    assert engine._rollouts_in_flight() is None
    assert engine._was_not_running_msg("arm") == "Was not running on 'arm'"


def test_the_robot_the_caller_named_is_never_reported_as_still_running(sim) -> None:
    """Called directly, in the shape the production race can produce.

    ``was_running`` is read from the same population, so through ``stop_policy``
    a robot in flight takes the "Stopped" branch. The window the MuJoCo verdict
    is commented against - the claim raised between the population read and the
    flag write - reaches this sentence with the named robot in flight, and
    "Was not running on 'beta'. A policy is running on 'beta'" is the one thing
    it must never say.
    """
    _start(sim, "beta")
    _await_in_flight(sim, "beta")
    try:
        assert sim._was_not_running_msg("beta") == "Was not running on 'beta'"
    finally:
        sim.stop_policy("beta")


def test_the_base_verdict_names_the_outcome_it_reached() -> None:
    """The base engine's own two verdicts, on a backend that reaches them.

    MuJoCo overrides ``stop_policy`` (it joins the rollout's Future), so its
    cells above never exercise the base sentence - and a mutant that answered
    "Was not running" for a stop that really halted a rollout survived every
    cell in ``tests/simulation`` before this one. Newton is a base-verdict
    backend whose robots carry the durable claim.
    """
    from strands_robots.simulation.models import SimRobot
    from tests.simulation.test_stop_policy_base_contract import _JOINTS, _newton_engine

    engine = _newton_engine(policy_running=True)
    engine._world.robots["spare"] = SimRobot(
        name="spare", urdf_path="arm.xml", data_config="so100", joint_names=list(_JOINTS)
    )
    # A stop aimed at the idle robot while 'arm' is mid-rollout.
    assert _text(engine.stop_policy("spare")) == (
        "Was not running on 'spare'. A policy is running on 'arm'. Stop it first: action='stop_policy'."
    )
    assert _text(engine.stop_policy("arm")) == "Stopped on 'arm'"
    assert _text(engine.stop_policy("arm")) == "Was not running on 'arm'"
