"""Regression: a raising episode-outcome criterion is reported, not propagated.

The eval loops call a caller-supplied outcome criterion after every applied
action - ``eval_policy``'s ``success_fn`` and a benchmark spec's ``is_success``
/ ``is_failure``. Every other per-step hook on those same paths already states
what a raise means: ``on_frame`` is best-effort telemetry (warn and continue),
``spec.on_step`` returns ``status="error"`` naming the hook, and ``run``'s
``stop_when`` is fatal with a message naming the step. The outcome criterion -
the one that decides the evaluation's headline ``success_rate`` - had no such
policy, and neither ``evaluate`` nor ``_evaluate_with_spec`` carried the
terminal handler ``run`` has, so a criterion that raised left the method as the
caller's own exception. Two consequences: the ``eval_policy`` action returned a
traceback instead of a structured result, and every episode already completed
was discarded with nothing naming the episode that broke it.

The controls pin the boundary. A ``bool()``-coercible verdict (including the
NumPy scalar that ``observation["x"] > 0.5`` returns, which is not an instance
of ``bool``) must keep working, a false verdict must still be a measured
episode, an absent criterion must still report ``success_measured=False``, and
the neighbouring ``on_step`` / ``on_frame`` / cooperative-stop policies must be
unchanged - the terminal handler must not turn best-effort telemetry fatal.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import (  # noqa: E402
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import CooperativeStop  # noqa: E402

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


class _Spec(BenchmarkProtocol):
    """Always-running spec whose verdict hooks are supplied per test."""

    max_steps = 4

    def __init__(self, *, success: Any = None, failure: Any = None, step: Any = None) -> None:
        self._success = success or (lambda sim: False)
        self._failure = failure or (lambda sim: False)
        self._step = step

    @property
    def supported_robots(self) -> list[str]:
        return ["arm1"]

    @property
    def default_robot(self) -> str:
        return "arm1"

    def on_episode_start(self, sim: Any, rng: random.Random) -> None:
        return None

    def on_step(self, sim: Any, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        if self._step is not None:
            self._step(sim)
        return StepInfo(reward=0.0)

    def is_success(self, sim: Any) -> bool:
        return self._success(sim)

    def is_failure(self, sim: Any) -> bool:
        return self._failure(sim)


@pytest.fixture
def sim(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    engine = Simulation(tool_name="crit", mesh=False)
    engine.create_world()
    added = engine.add_robot(name="arm1", urdf_path=str(xml_path))
    assert added["status"] == "success", added
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _json(result: dict) -> dict:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _text(result: dict) -> str:
    return "".join(b.get("text", "") for b in result["content"] if isinstance(b, dict))


def _eval(engine, **kwargs) -> dict:
    return engine.eval_policy(robot_name="arm1", policy_provider="mock", max_steps=4, control_frequency=20.0, **kwargs)


def _bench(engine, spec: _Spec, **kwargs) -> dict:
    name = "crit_bench"
    register_benchmark(name, spec)
    try:
        return engine.evaluate_benchmark(
            benchmark_name=name,
            robot_name="arm1",
            policy_provider="mock",
            control_frequency=20.0,
            **kwargs,
        )
    finally:
        try:
            unregister_benchmark(name)
        except Exception:
            # Best-effort teardown; never mask the real assertion failure.
            pass


def _raise(exc: BaseException):
    def _fn(_subject):
        raise exc

    return _fn


class TestARaisingOutcomeCriterionIsReported:
    def test_a_raising_success_fn_returns_an_error_result(self, sim):
        """The ``eval_policy`` action returns a structured error rather than
        letting the caller's own exception escape the tool result."""
        result = _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))
        assert result["status"] == "error", result

    def test_the_error_names_the_criterion_the_episode_and_the_step(self, sim):
        """The message locates the failure, so a criterion that breaks on a
        later episode is not reported as an anonymous traceback."""
        result = _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))
        text = _text(result)
        assert "success_fn" in text, text
        assert "episode 0" in text, text
        assert "step " in text, text
        assert "KeyError" in text, text

    def test_the_error_payload_reports_how_far_the_evaluation_got(self, sim):
        """``episodes_completed`` distinguishes "broke immediately" from "broke
        on the last episode" without reading the log."""
        result = _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))
        payload = _json(result)
        assert payload["episodes_completed"] == 0, payload
        assert payload["n_episodes"] == 3, payload
        assert payload["stopped_reason"] == "error", payload

    def test_a_raising_spec_is_success_returns_an_error_result(self, sim):
        """Same contract on the benchmark route, whose ``on_step`` sibling
        already reported its own failures this way."""
        result = _bench(sim, _Spec(success=_raise(ValueError("no body 'mug'"))), n_episodes=2)
        assert result["status"] == "error", result
        text = _text(result)
        assert "is_success" in text, text
        assert "ValueError" in text, text

    def test_a_raising_spec_is_failure_returns_an_error_result(self, sim):
        """``is_failure`` is evaluated on the same per-step chain as
        ``is_success`` and carries the same contract."""
        result = _bench(sim, _Spec(failure=_raise(ValueError("no body 'mug'"))), n_episodes=2)
        assert result["status"] == "error", result
        assert "is_failure" in _text(result), _text(result)


class TestTheWorkingCriterionPathsAreUnchanged:
    """Controls: these hold both before and after the fix. They fail if the
    coercion rejects a legitimate verdict or the terminal handler swallows a
    policy that another hook family already owns."""

    def test_a_true_verdict_still_reports_success(self, sim):
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: True)
        assert result["status"] == "success", result
        payload = _json(result)
        assert payload["success_rate"] == 1.0, payload
        assert payload["success_measured"] is True, payload

    def test_a_false_verdict_is_still_a_measured_episode(self, sim):
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: False)
        payload = _json(result)
        assert result["status"] == "success", result
        assert payload["success_rate"] == 0.0, payload
        assert payload["success_measured"] is True, payload

    def test_a_numpy_scalar_verdict_is_still_accepted(self, sim):
        """``observation["x"] > 0.5`` returns ``numpy.bool_``, which is NOT an
        instance of ``bool`` - so the verdict must be coerced, never
        type-checked."""
        np = pytest.importorskip("numpy")
        assert not isinstance(np.bool_(True), bool)  # premise
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: np.bool_(True))
        assert result["status"] == "success", result
        assert _json(result)["success_rate"] == 1.0, _json(result)

    def test_an_absent_criterion_is_still_reported_as_unmeasured(self, sim):
        result = _eval(sim, n_episodes=2)
        payload = _json(result)
        assert result["status"] == "success", result
        assert payload["success_measured"] is False, payload

    def test_a_raising_on_step_still_reports_its_own_message(self, sim):
        """The adjacent guard keeps its own wording; the new handler does not
        displace it."""
        result = _bench(sim, _Spec(step=_raise(RuntimeError("sensor gone"))), n_episodes=1)
        assert result["status"] == "error", result
        assert "on_step failed" in _text(result), _text(result)

    def test_a_raising_on_frame_hook_stays_best_effort(self, sim):
        """``on_frame`` is telemetry: a failure is logged and the eval still
        completes. A terminal handler that made it fatal would break this."""

        def boom(step, obs, action):
            raise RuntimeError("telemetry sink down")

        result = _eval(sim, n_episodes=2, success_fn=lambda obs: False, on_frame=boom)
        assert result["status"] == "success", result
        assert _json(result)["episodes_completed"] == 2, _json(result)

    def test_a_cooperative_stop_still_ends_the_eval_cleanly(self, sim):
        """``CooperativeStop`` is a BaseException and its handler precedes the
        terminal one, so a graceful stop still reports the completed episodes."""
        calls = {"n": 0}

        def stop_after_two(step, obs, action):
            calls["n"] += 1
            if calls["n"] > 2:
                raise CooperativeStop()

        result = _eval(sim, n_episodes=3, success_fn=lambda obs: False, on_frame=stop_after_two)
        assert result["status"] == "success", result
        assert _json(result)["stopped_early"] is True, _json(result)
