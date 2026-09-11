"""A caller-supplied ``on_frame`` hook must be callable on every eval surface.

``on_frame`` is the caller's own per-step lane on the eval routes: it fires after
every applied ``send_action`` and is where a caller attaches synchronous frame
recording or telemetry. An exception from it is deliberately best-effort - logged
at WARN, never fatal - because a hook that fails at frame 700 must not discard
699 good episodes.

That posture was applied to a value that was never callable at all, which is a
caller error knowable before the first step. Measured on a 6-step evaluation
whose hook is the integer ``42``:

    surface                          status    actions applied   hook ran   told the caller
    eval_policy(on_frame=42)         success   6                 never      nothing
    PolicyRunner.evaluate(...)       success   6                 never      nothing
    PolicyRunner.run(...)            error     5                 never      "silent dataset corruption"
    run_policy(observer=42)          error     0                 -          "must be callable or None"

The ``eval_policy`` envelope was byte-identical to the healthy call - same
success rate, same step counts - while the caller's telemetry recorded nothing
and the only trace was one repeated ``TypeError`` per frame in the log. The run
path did report an error, but only after five actions had already been applied,
and it blamed the dataset recorder for a value visible at the door.

``run_policy``'s ``observer`` lane already carried the domain (refused at the
facade and in ``PolicyRunner.run``, before any side effect). These tests pin the
same rule for the hook, on the values rather than on wording: every surface
refusing it before an action or an inference, the best-effort posture for a hook
that genuinely fails left intact, the refusal text being the shared helper's
verbatim, and a structural sweep so a new callback parameter cannot ship without
the domain.
"""

from __future__ import annotations

import ast
import inspect
import logging
import pathlib
from typing import Any

import pytest

import strands_robots.simulation.base as base_mod
import strands_robots.simulation.policy_runner as runner_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import optional_callable_error

from .test_policy_runner_async_rtc import _ChunkPolicy, _CountingSim

# Values no hook can be called from: an int, a string that names one, an empty
# tuple, a plain object, and a dict of hooks (the shape a caller reaches for when
# they mean to pass one of several).
UNCALLABLE: list[Any] = [42, "record_frame", (), object(), {"telemetry": print}]
_IDS = ["int", "str", "tuple", "object", "dict"]


def _arm() -> tuple[_CountingSim, _ChunkPolicy]:
    """A GL-free engine plus a policy that records every inference it is asked for."""
    sim = _CountingSim()
    policy = _ChunkPolicy(sim, infer_sleep=0.0)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


class TestEverySurfaceRefusesAHookItCannotCall:
    """The refusal reaches the caller, and costs no action and no inference."""

    @pytest.mark.parametrize("value", UNCALLABLE, ids=_IDS)
    def test_policy_runner_evaluate_refuses(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match=r"PolicyRunner\.evaluate: on_frame must be callable"):
            PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=6, on_frame=value)
        assert (sim.send_count, policy.infer_starts) == (0, [])

    @pytest.mark.parametrize("value", UNCALLABLE, ids=_IDS)
    def test_policy_runner_run_refuses(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match=r"PolicyRunner\.run: on_frame must be callable"):
            PolicyRunner(sim).run("arm", policy, n_steps=6, fast_mode=True, on_frame=value)
        assert (sim.send_count, policy.infer_starts) == (0, [])

    @pytest.mark.parametrize("value", UNCALLABLE, ids=_IDS)
    def test_eval_policy_answers_with_an_envelope(self, value: Any) -> None:
        sim, policy = _arm()
        result = sim.eval_policy(robot_name="arm", policy_object=policy, n_episodes=1, max_steps=6, on_frame=value)
        assert result["status"] == "error"
        assert "on_frame must be callable or None" in result["content"][0]["text"]
        assert (sim.send_count, policy.infer_starts) == (0, [])

    @pytest.mark.parametrize("value", UNCALLABLE, ids=_IDS)
    def test_evaluate_benchmark_answers_with_an_envelope(self, value: Any) -> None:
        sim, policy = _arm()
        result = sim.evaluate_benchmark(
            "any-benchmark", robot_name="arm", policy_object=policy, n_episodes=1, on_frame=value
        )
        assert result["status"] == "error"
        assert "on_frame must be callable or None" in result["content"][0]["text"]
        assert (sim.send_count, policy.infer_starts) == (0, [])

    def test_the_hook_shares_the_precedence_of_the_sibling_domains(self) -> None:
        """Parity, not a new ordering: caller domains answer before the lookup."""
        sim, policy = _arm()
        hook = sim.evaluate_benchmark("any-benchmark", policy_object=policy, on_frame=42)  # type: ignore[arg-type]
        episodes = sim.evaluate_benchmark("any-benchmark", policy_object=policy, n_episodes=0)
        assert "on_frame" in hook["content"][0]["text"]
        assert "n_episodes" in episodes["content"][0]["text"]


class TestTheBestEffortPostureSurvives:
    """Over-reach control: a hook that FAILS is still telemetry, not a caller error."""

    def test_a_raising_hook_is_still_absorbed_by_the_eval_loop(self, caplog: pytest.LogCaptureFixture) -> None:
        sim, policy = _arm()
        calls: list[int] = []

        def hook(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
            calls.append(step)
            raise RuntimeError("telemetry sink down")

        with caplog.at_level(logging.WARNING, logger=runner_mod.__name__):
            result = PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=6, on_frame=hook)

        assert result["status"] == "success"
        assert len(calls) == 6
        assert sum("on_frame hook failed" in record.message for record in caplog.records) == 6

    def test_a_raising_hook_still_trips_the_rollout_watchdog(self) -> None:
        sim, policy = _arm()

        def hook(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
            raise RuntimeError("telemetry sink down")

        result = PolicyRunner(sim).run("arm", policy, n_steps=20, fast_mode=True, on_frame=hook)
        assert result["status"] == "error"
        assert "aborting episode to avoid silent dataset corruption" in result["content"][0]["text"]

    @pytest.mark.parametrize("hook", [None, lambda step, obs, action: None])
    def test_a_usable_hook_still_evaluates(self, hook: Any) -> None:
        sim, policy = _arm()
        result = sim.eval_policy(robot_name="arm", policy_object=policy, n_episodes=1, max_steps=6, on_frame=hook)
        assert result["status"] == "success"
        assert sim.send_count == 6

    def test_the_hook_still_sees_every_applied_step(self) -> None:
        """Non-vacuity for the control above: a usable hook really does fire."""
        sim, policy = _arm()
        seen: list[int] = []
        result = sim.eval_policy(
            robot_name="arm",
            policy_object=policy,
            n_episodes=1,
            max_steps=6,
            on_frame=lambda step, obs, action: seen.append(step),
        )
        assert result["status"] == "success"
        assert seen == [0, 1, 2, 3, 4, 5]


class TestTheDomainIsTheObserverLanesVerbatim:
    """One shared rule, so a hook refused on one surface cannot pass on another."""

    @pytest.mark.parametrize("value", UNCALLABLE, ids=_IDS)
    def test_the_text_is_the_shared_helpers_own(self, value: Any) -> None:
        sim, policy = _arm()
        result = sim.eval_policy(robot_name="arm", policy_object=policy, n_episodes=1, max_steps=6, on_frame=value)
        assert result["content"][0]["text"] == optional_callable_error(value, "on_frame", "eval_policy")

    def test_the_parameter_default_is_inside_the_domain(self) -> None:
        for surface in (SimEngine.eval_policy, SimEngine.evaluate_benchmark, PolicyRunner.evaluate, PolicyRunner.run):
            default = inspect.signature(surface).parameters["on_frame"].default
            assert optional_callable_error(default, "on_frame", surface.__qualname__) is None


# ``stop_when`` is callable-annotated too and is deliberately NOT in the
# population: it is a two-shape early-return clause, refused at the facade by its
# own compiler ("expected a predicate call like {'predicate': 'grasped', ...}",
# before any action) and fatal on the first step for a direct runner caller,
# naming the parameter and why the rollout cannot continue. Pinned below so the
# exclusion is a measured fact rather than a hole in the sweep.
_OTHER_MECHANISM = {"stop_when"}


def _callback_params(func: Any) -> set[str]:
    """Per-step callback parameter names the CALLER supplies, by annotation."""
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if any(token in str(param.annotation) for token in ("Callable", "OnFrame", "RunPolicyObserver"))
    } - _OTHER_MECHANISM


def _guarded_params(module: Any, cls_name: str, method_name: str) -> set[str]:
    """Parameter names this method passes to ``optional_callable_error``."""
    tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text())
    func = next(
        m
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == cls_name
        for m in node.body
        if isinstance(m, ast.FunctionDef) and m.name == method_name
    )
    return {
        arg.id
        for call in ast.walk(func)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "optional_callable_error"
        for arg in call.args
        if isinstance(arg, ast.Name)
    }


class TestEveryCallerSuppliedCallbackOwnsTheDomain:
    """A structural sweep, so the next callback parameter cannot forget it."""

    SURFACES = (
        (base_mod, "SimEngine", "run_policy", {"observer"}),
        (base_mod, "SimEngine", "eval_policy", {"on_frame"}),
        (base_mod, "SimEngine", "evaluate_benchmark", {"on_frame"}),
        (runner_mod, "PolicyRunner", "run", {"observer", "on_frame"}),
        (runner_mod, "PolicyRunner", "evaluate", {"on_frame"}),
    )

    @pytest.mark.parametrize(("module", "cls_name", "method", "expected"), SURFACES)
    def test_the_discovered_callbacks_are_the_ones_guarded(
        self, module: Any, cls_name: str, method: str, expected: set[str]
    ) -> None:
        func = getattr(getattr(module, cls_name), method)
        assert _callback_params(func) == expected, "a callback parameter was added or renamed"
        assert _guarded_params(module, cls_name, method) == expected

    def test_the_sweep_detects_an_unguarded_surface(self) -> None:
        """Non-vacuity: the scanner must fail on a surface that skips the guard."""
        planted = ast.parse("class PolicyRunner:\n    def rollout(self, on_frame=None):\n        return on_frame\n")
        cls = planted.body[0]
        assert isinstance(cls, ast.ClassDef)
        func = cls.body[0]
        assert isinstance(func, ast.FunctionDef)
        assert not {
            arg.id
            for call in ast.walk(func)
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "optional_callable_error"
            for arg in call.args
            if isinstance(arg, ast.Name)
        }

    def test_the_excluded_clause_is_refused_by_its_own_mechanism(self) -> None:
        """Why ``stop_when`` may sit outside the sweep, measured on both layers."""
        sim, policy = _arm()
        facade = sim.run_policy(
            robot_name="arm",
            policy_object=policy,
            n_steps=6,
            fast_mode=True,
            stop_when=42,  # type: ignore[arg-type]
        )
        assert facade["status"] == "error"
        assert "stop_when" in facade["content"][0]["text"]
        assert sim.send_count == 0

        direct_sim, direct_policy = _arm()
        runner = PolicyRunner(direct_sim).run(
            "arm",
            direct_policy,
            n_steps=6,
            fast_mode=True,
            stop_when=42,  # type: ignore[arg-type]
        )
        assert runner["status"] == "error"
        assert "stop_when predicate raised at step 1" in runner["content"][0]["text"]

    def test_the_evaluate_guard_precedes_the_benchmark_delegation(self) -> None:
        """The spec route inherits it: ``_evaluate_with_spec`` is reached later."""
        tree = ast.parse(pathlib.Path(inspect.getfile(runner_mod)).read_text())
        func = next(
            m
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "PolicyRunner"
            for m in node.body
            if isinstance(m, ast.FunctionDef) and m.name == "evaluate"
        )
        guard_line = min(
            call.lineno
            for call in ast.walk(func)
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "optional_callable_error"
        )
        spec_line = min(
            call.lineno
            for call in ast.walk(func)
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "_evaluate_with_spec"
        )
        assert guard_line < spec_line
