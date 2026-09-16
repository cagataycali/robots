"""A policy_config the provider's constructor refuses is a result, on every rollout surface.

``_preflight_policy_config`` reports what can be judged without constructing: an
unresolvable provider name, and the provider's own ``preflight`` hook. What a
constructor judges for itself only exists once it runs -- ``Gr00tPolicy: invalid
port: 70000 (expected 1-65535)``, or a keyword no pre-construction screen can
judge -- and it used to escape differently on each surface. Measured on ``so101`` before this change:

* ``run_policy``, ``eval_policy`` and ``evaluate_benchmark`` raised the bare
  ``ValueError`` past the ``status=error`` envelope every sibling refusal (an
  unknown robot, no world, a busy robot) is returned as.
* ``start_policy`` builds the policy on its worker, so the raise lived in a
  Future nobody read. The call reported ``Policy started on 'so101' (async)``,
  ``list_policies_running`` then reported ``No policies running.`` -- the same
  reading as a rollout that ran to completion -- and nothing was logged. A
  rollout that never produced an action was indistinguishable from one that
  finished.
"""

from __future__ import annotations

import logging
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import factory as policy_factory
from strands_robots.policies import register_policy
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

# A port outside 1-65535: refused by Gr00tPolicy.__init__ itself, which is the
# point -- no screen that runs before construction can know this rule.
_BAD_PORT = {"host": "127.0.0.1", "port": 70000, "data_config": "so101"}
_PORT_VERDICT = "Gr00tPolicy: invalid port: 70000"
# A misspelling of a parameter the provider does read. Cosmos3Policy declares no
# **kwargs sink and its signature is readable, so ``policy_kwargs_error`` judges
# this keyword before anything is constructed and names the parameter meant.
_UNBOUND_KEYWORD = {"hots": "127.0.0.1"}
# The other arm. ``policy_kwargs_error`` screens against the constructor's own
# signature, so a constructor that binds nothing at all leaves it with an empty
# accepted set and nothing to screen -- the one route on which CPython's own
# TypeError still reaches this seam, and the reason it stays in the caught tuple.
_BINDS_NOTHING = "constructor_binds_nothing_probe"

_BENCHMARK = "constructor_refusal_probe"


class _BindsNothingPolicy(MockPolicy):
    """A provider whose constructor binds no keyword, so no screen can judge one."""

    def __init__(self) -> None:
        super().__init__()


def _text(result: dict) -> str:
    return result["content"][0]["text"]


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine()
    engine.create_world()
    engine.add_robot("so101")
    yield engine
    engine.cleanup()


@pytest.fixture
def binds_nothing_provider():
    """A registered provider whose constructor ``policy_kwargs_error`` cannot screen."""
    register_policy(_BINDS_NOTHING, lambda: _BindsNothingPolicy)
    try:
        yield _BINDS_NOTHING
    finally:
        policy_factory._runtime_registry.pop(_BINDS_NOTHING, None)


@pytest.fixture
def benchmark():
    """A trivially-satisfiable so101 benchmark, so only the policy can fail."""
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": _BENCHMARK,
            "max_steps": 4,
            "supported_robots": ["so101"],
            "default_robot": "so101",
        }
    )
    register_benchmark(bench.name, bench)
    yield bench.name
    unregister_benchmark(bench.name)


def _await_idle(sim: MuJoCoSimEngine, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sim._rollouts_in_flight():
            return
        time.sleep(0.05)
    raise AssertionError("rollout still in flight")


def _await_record(sim: MuJoCoSimEngine, timeout: float = 2.0) -> None:
    """The done-callback runs on the worker, just after the Future completes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not sim._rollouts_ended_in_error():
        time.sleep(0.02)


class TestTheBlockingSurfaces:
    """Each entry point returns the refusal it used to raise, and names itself."""

    def test_run_policy_reports_it(self, sim):
        result = sim.run_policy(robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, duration=0.2)
        assert result["status"] == "error"
        assert _text(result).startswith(
            f"run_policy: policy provider 'groot' refused its configuration, so no rollout was started. {_PORT_VERDICT}"
        )

    def test_eval_policy_reports_it(self, sim):
        result = sim.eval_policy(
            robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, n_episodes=1, max_steps=2
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            f"eval_policy: policy provider 'groot' refused its configuration, "
            f"so no rollout was started. {_PORT_VERDICT}"
        )

    def test_evaluate_benchmark_reports_it(self, sim, benchmark):
        result = sim.evaluate_benchmark(
            benchmark, robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, n_episodes=1
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            f"evaluate_benchmark: policy provider 'groot' refused its configuration, "
            f"so no rollout was started. {_PORT_VERDICT}"
        )

    def test_a_keyword_the_screen_can_judge_names_the_parameter_meant(self, sim):
        """The screened arm: the refusal is reached without constructing anything.

        ``create_policy`` screens the resolved kwargs against the constructor's
        own signature first (``policy_kwargs_error``), so for a provider whose
        signature it can read the report names the parameter the keyword
        misspells rather than only the keyword CPython would have named.
        """
        result = sim.run_policy(
            robot_name="so101", policy_provider="cosmos3", policy_config=_UNBOUND_KEYWORD, duration=0.2
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            "run_policy: policy provider 'cosmos3' refused its configuration, so no rollout was started."
        )
        # The keyword is named, and so is the parameter no other name could mean.
        assert "does not accept 'hots' (did you mean 'host'?)" in _text(result)

    def test_a_keyword_no_screen_can_judge_is_the_same_envelope(self, sim, binds_nothing_provider):
        """The ``TypeError`` arm: the keyword reaches the constructor and CPython refuses it.

        A ``ValueError`` the constructor raises on purpose and CPython's own
        ``TypeError`` for an unbound keyword reach this seam by different
        routes, and both used to leave the library as a traceback. The screen
        closed the route every registered provider took to get here -- each one
        has a signature it can read -- so the arm is measured on the case it
        still cannot judge, which is what keeps ``TypeError`` in the caught
        tuple honest rather than defensive.
        """
        result = sim.run_policy(
            robot_name="so101",
            policy_provider=binds_nothing_provider,
            policy_config=_UNBOUND_KEYWORD,
            duration=0.2,
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            f"run_policy: policy provider {binds_nothing_provider!r} refused its configuration, "
            "so no rollout was started."
        )
        # The constructor's own words, not a paraphrase: the keyword is named.
        assert "unexpected keyword argument 'hots'" in _text(result)

    def test_the_trust_gate_keeps_its_raise(self, sim):
        """Not every refusal belongs in this envelope, and the boundary is the exception type.

        The remote-code gate names its remedy already and holds for every call
        in the process rather than for this configuration, so it must keep
        travelling as an exception. It is a ``RuntimeError``, which is what
        keeps it outside ``(TypeError, ValueError)``; widening that tuple to
        ``Exception`` would silently turn a security decision into an
        error string a caller may retry past.
        """
        from strands_robots.policies.factory import UntrustedRemoteCodeError

        assert not issubclass(UntrustedRemoteCodeError, TypeError | ValueError)
        with pytest.raises(UntrustedRemoteCodeError):
            sim.run_policy(robot_name="so101", policy_provider="kimodo", duration=0.2)

    def test_the_robot_is_left_free(self, sim):
        """A refusal before construction consumed no rollout slot."""
        sim.run_policy(robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, duration=0.2)
        assert _text(sim.list_policies_running()) == "No policies running."
        ok = sim.run_policy(robot_name="so101", policy_provider="mock", duration=0.1)
        assert ok["status"] == "success"


class TestTheAsynchronousSurface:
    """``start_policy`` answers before the worker builds anything, so the worker reports."""

    def test_a_constructor_refusal_on_the_worker_is_reported_where_a_caller_reads(self, sim, caplog):
        with caplog.at_level(logging.ERROR, logger="strands_robots.simulation.mujoco.simulation"):
            started = sim.start_policy(
                robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, duration=1.0
            )
            # Still a success: the verdict is given before the worker exists,
            # and inventing a synchronous build here would make start_policy
            # block on the download it exists to avoid.
            assert started["status"] == "success"
            _await_idle(sim)
            _await_record(sim)
        listed = _text(sim.list_policies_running())
        assert listed.startswith(
            "No policies running.\nRollouts started with start_policy that ended in error (1):\n  - so101: "
        )
        assert _PORT_VERDICT in listed
        assert any("start_policy rollout on 'so101'" in rec.getMessage() for rec in caplog.records)

    def test_a_refusal_that_keeps_its_raise_is_still_reported_on_the_worker(self, sim, caplog):
        """The other arm of the done-callback: ``future.exception()``, not a result.

        The trust-remote-code gate and a missing optional dependency
        deliberately keep raising (see ``test_the_trust_gate_keeps_its_raise``),
        which on the blocking surfaces reaches the caller. On ``start_policy``
        it reached nobody: the raise stayed in the Future. Both routes have to
        land in the same reading, so the reason is prefixed with the exception
        type the caller would otherwise have caught.
        """
        with caplog.at_level(logging.ERROR, logger="strands_robots.simulation.mujoco.simulation"):
            assert sim.start_policy(robot_name="so101", policy_provider="kimodo", duration=1.0)["status"] == "success"
            _await_idle(sim)
            _await_record(sim)
        listed = _text(sim.list_policies_running())
        assert "Rollouts started with start_policy that ended in error (1):" in listed
        assert "so101: UntrustedRemoteCodeError: " in listed
        assert "STRANDS_TRUST_REMOTE_CODE=1" in listed  # the remedy survives the trip
        assert any("start_policy rollout on 'so101'" in rec.getMessage() for rec in caplog.records)

    def test_a_robot_a_rollout_is_still_driving_is_not_also_listed_as_failed(self, sim):
        """A robot cannot be both readings at once.

        The record is cleared when that robot's next rollout is submitted, but
        a done-callback runs after its Future is marked done, so it can land
        just after a second rollout was admitted. Seeded here directly, since
        that ordering is not something a test can schedule.
        """
        assert sim.start_policy(robot_name="so101", policy_provider="mock", duration=2.0)["status"] == "success"
        sim._rollout_failures["so101"] = "a verdict from the rollout before this one"
        listed = _text(sim.list_policies_running())
        assert listed == "Active policies (1):\n  - so101"
        sim.stop_policy(robot_name="so101")
        _await_idle(sim)

    def test_the_next_rollout_on_that_robot_replaces_the_record(self, sim):
        """The record is the last outcome, not a log: a later rollout clears it."""
        sim.start_policy(robot_name="so101", policy_provider="groot", policy_config=_BAD_PORT, duration=1.0)
        _await_idle(sim)
        _await_record(sim)
        assert sim._rollouts_ended_in_error() != {}
        assert sim.start_policy(robot_name="so101", policy_provider="mock", duration=0.2)["status"] == "success"
        _await_idle(sim)
        time.sleep(0.1)
        assert _text(sim.list_policies_running()) == "No policies running."

    def test_a_clean_rollout_leaves_the_idle_reading_untouched(self, sim):
        assert sim.start_policy(robot_name="so101", policy_provider="mock", duration=0.2)["status"] == "success"
        _await_idle(sim)
        time.sleep(0.1)
        assert _text(sim.list_policies_running()) == "No policies running."


def test_the_base_seam_defaults_to_no_record():
    """A backend with no asynchronous entry has nothing to add to the reading."""
    from strands_robots.simulation.base import SimEngine

    assert SimEngine._rollouts_ended_in_error(object()) == {}  # type: ignore[arg-type]
