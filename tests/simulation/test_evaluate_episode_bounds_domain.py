"""Accepted domain of the two bounds of ``PolicyRunner.evaluate``'s episode loop.

``n_episodes`` and ``max_steps`` are the outer and inner bound of the nested loop
``evaluate`` runs. They applied no domain, and unlike a knob whose misuse degrades
an evaluation these two *remove* it while still reporting one:

* ``success_rate`` is ``n_success / max(n_completed, 1)``, so the guard that
  protects the division also turns "nothing ran" into a clean ``0.0``, and
* ``success_measured`` is ``resolved_check is not None`` - derived from whether a
  criterion was *supplied*, not from whether it was ever evaluated.

So ``n_episodes=0`` and ``max_steps=0`` each returned ``status="success"`` with
``success_rate: 0.0`` **and** ``success_measured: true`` over zero applied
actions - a payload indistinguishable from a policy that genuinely failed every
episode, and ``success_measured`` is the field that exists precisely so a ``0.0``
cannot be mistaken for a measurement. ``max_steps=nan`` reached the same result
silently (``steps < nan`` is ``False`` on the first test), ``2.7`` and ``True``
truncated to a horizon nobody typed, and ``max_steps=inf`` did not report a wrong
number at all: ``while steps < max_steps`` has no false case, so the first
episode never returned (measured at ~55k control steps per second against a
no-op sim; against a real policy each step is a forward pass).

The same method already refuses the same mistake one path over.
:meth:`PolicyRunner._evaluate_with_spec` checks the *benchmark's* horizon with
``positive_count_error(spec.max_steps, "max_steps", "evaluate_benchmark")`` and a
message describing exactly what ``evaluate`` did with its own parameter - "a
non-positive one runs episodes of zero length and reports a 0% success rate over
them instead of surfacing the mistake" - and its comment explains the placement
by asserting that "every other bound of this nested loop (``n_episodes``,
``action_horizon``, ``control_substeps``) is checked by the public entry point
before it gets here". That premise holds for the facade and not for this layer:
``SimEngine.eval_policy`` does check both, but ``PolicyRunner`` is documented as
drivable directly and for a direct caller the parenthetical was false.

The rule this layer states for itself, verbatim from ``_control_substeps``: "The
public entry points reject such a value with a structured error before reaching
the runner; **this raise is the guarantee for callers driving ``PolicyRunner``
directly.**" ``evaluate`` already honored it for ``seed``, ``action_horizon``
(#2080), ``rtc_inference_timeout_s``, ``control_substeps`` and the recording
rate. These two were the remainder.

These tests pin, on the values rather than on wording:

* both bounds refused over the whole shared domain, naming parameter and method,
* the refusal costing no reset, no inference, no action and no reseed of the
  process-global RNG,
* the fabricated ``success_measured: true`` payload being unreachable,
* ``max_steps=inf`` refused rather than run - bounded by a fuse in the stub sim
  rather than by a timeout, so a regression fails instead of hanging,
* usable bounds still running with their step accounting unchanged,
* ``max_steps`` scoped to the path that reads it (with a ``spec`` it is
  documented as ignored, and the spec's own horizon is refused where it is read),
* and the domain being the entry point's verbatim.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import random
from typing import Any

import numpy as np
import pytest

import strands_robots.simulation.policy_runner as runner_mod
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import positive_count_error

from .test_policy_runner_async_rtc import _CountingSim
from .test_policy_runner_benchmark import FakeSim, _CountingBenchmark

# The shared domain's whole refused set, spelled as this module's sibling
# ``test_action_horizon_domain`` spells it: ``0``/negatives/``False`` because a
# loop cannot run fewer than one iteration; ``True`` because ``bool`` is an
# ``int`` subclass and acted as a silent count of 1; ``2.7`` because a fractional
# bound truncates to one nobody typed; ``3.0`` and the numpy spellings because
# these values are consumed as ``range()`` bounds and comparison operands where
# only a true ``int`` can be honored; ``nan``/``inf``/``None``/a list/a dict
# because no comparison against them terminates the loop or because the loop
# cannot be built from them at all.
UNUSABLE: list[Any] = [
    0,
    -5,
    -1,
    True,
    False,
    2.7,
    3.0,
    math.nan,
    math.inf,
    "8",
    None,
    [4],
    {},
    np.int64(3),
    np.float64(4.0),
]

# Small so the accounting assertions stay exact and cheap: an accepted pair runs
# ``n_episodes * max_steps`` control steps.
USABLE: list[int] = [1, 2, 3]

BOUNDS = ["n_episodes", "max_steps"]


def _ident(value: Any) -> str:
    """A stable label; ``0``/``False`` and ``1``/``True`` hash equal."""
    if value is True:
        return "True"
    if value is False:
        return "False"
    return repr(value)


class _FuseBlown(RuntimeError):
    """Raised by the stub sim when a rollout exceeds a bound no test asks for."""


class _EvalSim(_CountingSim):
    """``_CountingSim`` plus a reset counter and a runaway fuse.

    The fuse is what makes the ``max_steps=inf`` case testable without a timeout
    and without a thread: on the pre-fix tree that rollout never terminates, so
    an unbounded stub would hang the suite. Raising here converts a regression
    into a failure with a message, and the fuse's own non-vacuity is pinned by
    :meth:`TestTheInfiniteHorizonIsRefusedRatherThanRun.test_the_fuse_really_fires`.
    """

    fuse: int = 5_000

    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    def reset(self) -> dict[str, Any]:
        self.reset_count += 1
        return {"status": "success"}

    def send_action(self, action: Any, robot_name: Any = None, n_substeps: int = 1) -> Any:
        if self.send_count >= self.fuse:
            raise _FuseBlown(
                f"the rollout applied {self.send_count} actions without terminating; "
                "an episode bound outside the accepted domain was not refused"
            )
        return super().send_action(action, robot_name, n_substeps)


class _CountingPolicy(MockPolicy):
    """``MockPolicy`` that records how many times it was queried."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.infer_calls = 0

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.infer_calls += 1
        return await super().get_actions(observation_dict, instruction, **kwargs)


def _arm() -> tuple[_EvalSim, _CountingPolicy]:
    """A GL-free engine plus a policy that records its calls."""
    sim = _EvalSim()
    policy = _CountingPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


def _evaluate(sim: Any, policy: Any, **kwargs: Any) -> dict[str, Any]:
    """``evaluate`` with a success criterion genuinely in force.

    ``success_fn`` is supplied and always ``False``, so ``success_measured`` is
    ``True`` for the right reason and a ``success_rate`` of ``0.0`` is the value
    under test rather than the documented no-criterion placeholder.
    """
    return PolicyRunner(sim).evaluate(
        "arm",
        policy,
        success_fn=lambda obs: False,
        control_frequency=50.0,
        **kwargs,
    )


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return dict(result["content"][-1]["json"])


class TestBothEpisodeBoundsAreRefused:
    """Neither bound of the episode loop may take a value the loop cannot run."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=_ident)
    def test_n_episodes_is_refused(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match="n_episodes"):
            _evaluate(sim, policy, n_episodes=value, max_steps=3)

    @pytest.mark.parametrize("value", UNUSABLE, ids=_ident)
    def test_max_steps_is_refused(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match="max_steps"):
            _evaluate(sim, policy, n_episodes=2, max_steps=value)

    @pytest.mark.parametrize("param", BOUNDS)
    def test_the_message_names_the_parameter_and_the_method(self, param: str) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError) as exc:
            _evaluate(sim, policy, **{"n_episodes": 2, "max_steps": 3, param: 0})
        assert f"PolicyRunner.evaluate: {param}" in str(exc.value)


class TestTheRefusalCostsNothing:
    """A bound the loop cannot run must cost no episode, no model call, no reseed.

    Every one of these was paid on the pre-fix tree for ``max_steps`` - ``0`` /
    ``-5`` / ``nan`` reset the sim ``n_episodes`` times and returned a completed
    evaluation - and ``seed`` reseeds the process-global RNG for the whole
    process, so a refused eval that had already reseeded would leave a side
    effect behind on a call that did nothing.
    """

    @pytest.mark.parametrize("param", BOUNDS)
    @pytest.mark.parametrize("value", [0, -5, math.nan, math.inf, 2.7, None], ids=_ident)
    def test_no_reset_no_inference_no_action(self, param: str, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            _evaluate(sim, policy, **{"n_episodes": 2, "max_steps": 3, param: value})
        assert sim.reset_count == 0
        assert sim.send_count == 0
        assert policy.infer_calls == 0

    @pytest.mark.parametrize("param", BOUNDS)
    def test_the_process_global_rng_is_not_reseeded(self, param: str) -> None:
        sim, policy = _arm()
        random.seed(20260809)
        before = random.getstate()
        with pytest.raises(ValueError):
            _evaluate(sim, policy, **{"n_episodes": 2, "max_steps": 3, "seed": 4321, param: 0})
        assert random.getstate() == before

    def test_a_usable_pair_with_a_seed_really_does_reseed(self) -> None:
        """Non-vacuity: the state above is preserved because we refused, not
        because this eval path never reseeds."""
        sim, policy = _arm()
        random.seed(20260809)
        before = random.getstate()
        assert _evaluate(sim, policy, n_episodes=1, max_steps=1, seed=4321)["status"] == "success"
        assert random.getstate() != before


class TestTheFabricatedMeasurementIsUnreachable:
    """No bound may produce a ``success_measured`` payload over nothing.

    The acceptance criterion of #2082, asserted over the whole refused set on
    both parameters rather than on the two values that happened to be reported.
    """

    @pytest.mark.parametrize("param", BOUNDS)
    @pytest.mark.parametrize("value", UNUSABLE, ids=_ident)
    def test_no_payload_is_produced_at_all(self, param: str, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            _evaluate(sim, policy, **{"n_episodes": 2, "max_steps": 3, param: value})

    def test_a_measured_zero_still_means_a_policy_that_failed(self) -> None:
        """The flag keeps its meaning for a rollout that really ran.

        ``success_measured`` is derived from whether a criterion was supplied, so
        it cannot itself distinguish "nothing ran" - which is why the bounds have
        to be refused rather than the flag made cleverer.
        """
        sim, policy = _arm()
        payload = _payload(_evaluate(sim, policy, n_episodes=2, max_steps=3))
        assert payload["success_measured"] is True
        assert payload["success_rate"] == 0.0
        assert payload["episodes_completed"] == 2
        assert sim.send_count == 6

    def test_no_criterion_still_reports_unmeasured(self) -> None:
        """The neighbouring documented case is untouched by the guards."""
        sim, policy = _arm()
        result = PolicyRunner(sim).evaluate(
            "arm", policy, n_episodes=1, max_steps=2, success_fn=None, control_frequency=50.0
        )
        assert _payload(result)["success_measured"] is False


class TestTheInfiniteHorizonIsRefusedRatherThanRun:
    """``max_steps=inf`` was not a wrong number - it was a rollout with no end."""

    def test_the_loop_condition_has_no_false_case(self) -> None:
        """The premise, bounded and without running the loop."""
        assert all(steps < math.inf for steps in (0, 1, 300, 10**9))

    def test_evaluate_refuses_it_and_applies_no_action(self) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match="max_steps"):
            _evaluate(sim, policy, n_episodes=1, max_steps=math.inf)
        assert sim.send_count == 0
        assert policy.infer_calls == 0

    def test_the_fuse_really_fires(self) -> None:
        """Non-vacuity of the bound the test above relies on.

        Without this, a stub that silently stopped stepping would make the
        ``inf`` assertion pass for the wrong reason.
        """
        sim, policy = _arm()
        with pytest.raises(_FuseBlown):
            _evaluate(sim, policy, n_episodes=1, max_steps=_EvalSim.fuse + 1)


class TestUsableBoundsStillRun:
    """Over-reach control: the guards refuse exactly the unusable values."""

    @pytest.mark.parametrize("n_episodes", USABLE)
    @pytest.mark.parametrize("max_steps", USABLE)
    def test_the_step_accounting_is_unchanged(self, n_episodes: int, max_steps: int) -> None:
        sim, policy = _arm()
        payload = _payload(_evaluate(sim, policy, n_episodes=n_episodes, max_steps=max_steps))
        assert payload["episodes_completed"] == n_episodes
        assert payload["avg_steps"] == float(max_steps)
        assert sim.reset_count == n_episodes
        assert sim.send_count == n_episodes * max_steps

    def test_the_documented_defaults_are_inside_the_domain(self) -> None:
        """A default outside the domain would refuse every unparameterized call."""
        params = inspect.signature(PolicyRunner.evaluate).parameters
        defaults = {name: params[name].default for name in BOUNDS}
        assert defaults == {"n_episodes": 10, "max_steps": 300}
        for name, value in defaults.items():
            assert positive_count_error(value, name, "PolicyRunner.evaluate") is None


class TestMaxStepsIsScopedToThePathThatReadsIt:
    """With a ``spec``, ``max_steps`` is documented as ignored - so it is not refused.

    ``spec.max_steps`` wins there and is refused at its own read, because a
    benchmark's horizon has no parameter of its own to validate. Refusing the
    inert parameter as well would reject a value that changes nothing about the
    rollout - the same scoping ``SimEngine.run_policy`` applies to ``duration``
    ("validate the value the rollout will actually run on, and only then").
    ``n_episodes`` bounds the episode loop on both paths, so it is unconditional.
    """

    @staticmethod
    def _spec_arm() -> tuple[FakeSim, MockPolicy, _CountingBenchmark]:
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        return sim, policy, _CountingBenchmark()

    @pytest.mark.parametrize("value", [0, -5, math.nan, math.inf, 2.7, None], ids=_ident)
    def test_an_ignored_max_steps_is_not_refused(self, value: Any) -> None:
        sim, policy, spec = self._spec_arm()
        result = PolicyRunner(sim).evaluate(
            "fake_robot", policy, spec=spec, n_episodes=1, max_steps=value, control_frequency=50.0
        )
        assert result["status"] == "success"
        # The spec's horizon is the one that ran, not the parameter's.
        assert _payload(result)["max_steps"] == _CountingBenchmark.max_steps

    def test_the_specs_own_horizon_is_still_refused(self) -> None:
        """Non-vacuity of the scoping: the spec path has its own guard."""
        sim, policy, spec = self._spec_arm()
        spec.max_steps = 0  # type: ignore[misc]
        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1, control_frequency=50.0)
        assert result["status"] == "error"
        assert "max_steps" in result["content"][0]["text"]

    @pytest.mark.parametrize("value", [0, -5, math.inf, None], ids=_ident)
    def test_n_episodes_is_refused_on_the_spec_path_too(self, value: Any) -> None:
        sim, policy, spec = self._spec_arm()
        with pytest.raises(ValueError, match="n_episodes"):
            PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=value, control_frequency=50.0)


class TestTheDomainIsTheEntryPointsVerbatim:
    """A count refused for an eval through the facade cannot be accepted here.

    ``SimEngine.eval_policy`` delegates both bounds to
    :meth:`SimEngine._validate_positive_int`, which is the same shared
    ``positive_count_error``. Pinned as an equivalence over the values so the two
    layers cannot drift apart silently.
    """

    @pytest.mark.parametrize("param", BOUNDS)
    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=_ident)
    def test_the_runner_refuses_exactly_what_the_entry_point_refuses(self, param: str, value: Any) -> None:
        facade_refuses = SimEngine._validate_positive_int(value, param, "eval_policy") is not None
        sim, policy = _arm()
        try:
            _evaluate(sim, policy, **{"n_episodes": 1, "max_steps": 1, param: value})
        except ValueError:
            runner_refuses = True
        else:
            runner_refuses = False
        assert runner_refuses is facade_refuses

    @pytest.mark.parametrize("param", BOUNDS)
    def test_the_wording_is_the_shared_rule_with_this_surface_as_the_context(self, param: str) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError) as exc:
            _evaluate(sim, policy, **{"n_episodes": 1, "max_steps": 1, param: 0})
        assert str(exc.value) == positive_count_error(0, param, "PolicyRunner.evaluate")


class TestEveryEpisodeLoopBoundOwnsTheDomain:
    """Structural: a bound added to this signature must not forget the domain.

    The prose reason lives in this module's docstring; this is the part a future
    change cannot read. Keyed on the parameters ``evaluate`` validates by name.
    """

    EXPECTED = {"action_horizon", "n_episodes", "max_steps"}

    @staticmethod
    def _guarded_params(source: str, method: str) -> set[str]:
        """Parameter names passed to ``positive_count_error`` inside ``method``."""
        tree = ast.parse(source)
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method:
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "positive_count_error"
                        and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant)
                    ):
                        found.add(str(call.args[1].value))
        return found

    def test_each_bound_reaches_the_shared_domain(self) -> None:
        source = pathlib.Path(inspect.getfile(runner_mod)).read_text()
        assert self._guarded_params(source, "evaluate") == self.EXPECTED

    def test_the_sweep_detects_a_dropped_guard(self) -> None:
        """Meta-test: the assertion above fails when a guard is removed."""
        planted = (
            "class PolicyRunner:\n"
            "    def evaluate(self, n_episodes=10, max_steps=300, action_horizon=8):\n"
            "        if e := positive_count_error(action_horizon, 'action_horizon', 'x'):\n"
            "            raise ValueError(e)\n"
            "        if e := positive_count_error(n_episodes, 'n_episodes', 'x'):\n"
            "            raise ValueError(e)\n"
        )
        assert self._guarded_params(planted, "evaluate") == {"action_horizon", "n_episodes"}
        assert self._guarded_params(planted, "evaluate") != self.EXPECTED
