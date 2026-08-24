"""Accepted domain of ``PolicyRunner.run``'s horizon pair, and how the pair resolves.

``run`` takes the rollout length as *two* knobs. ``n_steps`` is an explicit step
count; ``duration`` is wall-clock seconds turned into ``int(duration *
control_frequency)`` steps. Only one of them can set the horizon, and the runner
picked which with::

    if n_steps is not None and n_steps > 0:   # <- the ``> 0``
        total_steps = int(n_steps)
    else:
        total_steps = int(duration * control_frequency)

so a step count outside its domain did not fail - it silently handed the horizon
to the *other* knob, whose default is ``10.0`` seconds. Measured on a GL-free
engine at ``control_frequency=50.0``, counting the ``send_action`` calls and
policy inferences the runner actually made:

* ``n_steps=0`` / ``-5`` / ``nan`` ran **500** steps, 500 applied actions and 126
  inferences - not a clamp to 1, and not the value the caller typed, but a
  horizon from a parameter they never set. ``2.7`` ran 2 and ``True`` ran 1.
* ``duration=0`` / ``0.0`` / ``-5`` / ``-0.5`` returned ``status="success"`` with
  ``steps_used=0`` and ``stopped_reason="budget"`` - the field a caller reads to
  decide whether to retry, asserting the horizon was exhausted when there was
  none - having applied no action at all. ``True`` ran a silent 1 second.
* ``nan`` / ``inf`` / ``"10"`` / ``[1.0]`` / ``None`` on either knob leaked a bare
  conversion or operand error ("cannot convert float NaN to integer", "can't
  multiply sequence by non-int of type 'float'") naming neither the parameter nor
  this method.

The pair therefore had to be resolved before anything could be refused, and the
tree had already resolved it in two places:

* :meth:`SimEngine._resolve_horizon`, the facade's resolver, applies
  :func:`~strands_robots.utils.positive_count_error` to ``n_steps`` whenever it is
  **not None** - not when it is ``> 0`` - so no public entry point can hand this
  runner an ``n_steps=0`` at all, and the ``> 0`` fallback was unreachable through
  every documented path.
* ``run``'s own ``Args:`` entry for ``duration`` says "Used only when ``n_steps``
  is None", which is that same rule stated on the parameter it governs.

So ``n_steps`` is effective whenever it is given, and ``duration`` is consulted
only when no step count was: the runner's rule is the facade's rule, which is
what the sibling knobs of this signature already do. ``action_horizon``,
``control_substeps``, ``control_frequency``, ``seed``,
``rtc_inference_timeout_s`` and ``max_onframe_failures`` each carry the entry
point's domain here as well, because ``PolicyRunner`` is documented as drivable
directly; ``_control_substeps`` states the reason verbatim - "this raise is the
guarantee for callers driving ``PolicyRunner`` directly". The horizon was the
last knob of the signature with no guarantee, and the only one whose value could
come from a parameter the caller did not set.

These tests pin, on the values rather than on wording:

* the premise - that the facade's resolver really refuses a step count on the
  "not None" condition, so the rule below is its rule and not a new one,
* every unusable value of both knobs refused, naming the parameter and the method,
* the refusal landing before any inference, any applied action, and before the
  process-global reseed ``seed=`` performs,
* ``n_steps=0`` no longer selecting ``duration``'s value,
* usable horizons still running, with the step accounting unchanged,
* ``duration`` judged only when it is the horizon, so a value the rollout will
  not read is not reported on - the same asymmetry the facade pins,
* the domain being the entry point's verbatim, including its numpy narrowing,
* and a structural sweep so a future runner surface cannot forget the domain.
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

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import positive_count_error, positive_finite_number_error

from .test_policy_runner_async_rtc import _ChunkPolicy, _CountingSim

# Step counts no horizon can be built from. ``0``/negatives because a rollout
# cannot run fewer than one step; ``True``/``False`` because ``bool`` is an
# ``int`` subclass and would act as a silent horizon of 1 (or fall through);
# ``2.7``/``3.0`` because a fractional or float horizon is not the declared type
# and truncates to one nobody typed; ``nan``/``inf``/``None``-adjacent shapes
# because the arithmetic cannot take them. The numpy spellings are refused by
# the entry point's own strict-int domain, so they are parity rather than a new
# narrowing - see ``TestTheDomainIsTheEntryPointsVerbatim``.
UNUSABLE_STEPS: list[Any] = [
    0,
    -1,
    -5,
    True,
    False,
    2.7,
    3.0,
    math.nan,
    math.inf,
    "8",
    [8],
    {},
    np.int64(3),
    np.float64(4.0),
]

# Durations no horizon can be built from. ``0``/negatives yield zero steps;
# ``True`` is a silent one second; ``nan``/``inf`` never survive the conversion;
# a string or a list cannot be multiplied by a frequency. ``None`` is included
# because ``duration`` has no "not supplied" spelling - it defaults to ``10.0``.
UNUSABLE_DURATION: list[Any] = [
    0,
    0.0,
    -5,
    -0.5,
    True,
    False,
    math.nan,
    math.inf,
    "10",
    [1.0],
    None,
    {},
]

# Values each knob CAN be built from.
USABLE_STEPS: list[int] = [1, 2, 8, 30]
USABLE_DURATION: list[float] = [0.02, 0.1, 1.0]


def _ident(value: Any) -> str:
    """A stable label; ``0``/``False`` and ``1``/``True`` hash equal."""
    if value is True:
        return "True"
    if value is False:
        return "False"
    return repr(value)


class _Counting(_ChunkPolicy):
    """Chunk policy that also records how many inferences it was asked for."""

    def __init__(self, sim: _CountingSim) -> None:
        super().__init__(sim, infer_sleep=0.0)
        self.infer_calls = 0

    async def get_actions(  # type: ignore[override]
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, float]]:
        self.infer_calls += 1
        return await super().get_actions(observation_dict, instruction, **kwargs)


def _arm() -> tuple[_CountingSim, _Counting]:
    """A GL-free engine plus a policy that records its own inference count."""
    sim = _CountingSim()
    policy = _Counting(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


def _run(**kwargs: Any) -> tuple[dict[str, Any], _CountingSim, _Counting]:
    """Drive ``PolicyRunner.run`` with deliberately un-narrowed horizon values.

    The horizon knobs are handed values outside their declared types, which is
    the whole point of measuring them; funnelled through one annotated helper so
    that intention is stated once instead of suppressed per call.
    """
    sim, policy = _arm()
    result = PolicyRunner(sim).run("arm", policy, control_frequency=50.0, action_horizon=1, **kwargs)
    return result, sim, policy


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """The structured json block of a rollout result."""
    return next(b["json"] for b in result["content"] if "json" in b)


def _facade_resolve(n_steps: Any) -> tuple[float, Any, dict[str, Any] | None]:
    """``SimEngine._resolve_horizon`` with a deliberately un-narrowed step count.

    The premise tests hand it values outside the declared ``int | None`` - which
    is the whole point of measuring what the facade does with them. Funnelled
    through one annotated helper so that intention is stated once instead of
    suppressed per call.
    """
    return SimEngine._resolve_horizon(n_steps, None, 50.0, 10.0, "run_policy")


class TestThePairResolutionIsTheFacadesRule:
    """The premise: the rule enforced below is the one the tree already had.

    Without this the domain would rest on a reading of the pair rather than on
    the resolver and docstring that already state it, and a future change to
    either could make the runner's rule and the facade's diverge silently.
    """

    @pytest.mark.parametrize("value", [0, -5, 2.7, True], ids=_ident)
    def test_the_facade_refuses_a_step_count_on_the_not_none_condition(self, value: Any) -> None:
        """``_resolve_horizon`` judges ``n_steps`` whenever it is given.

        Not when it is ``> 0`` - so no public entry point can hand this runner a
        step count outside the domain, and the ``> 0`` fallback the runner used
        was unreachable through every documented path.
        """
        _duration, _steps, error = _facade_resolve(value)
        assert error is not None, f"the facade accepted n_steps={value!r}"
        assert "n_steps" in error["content"][0]["text"]

    def test_the_facade_falls_back_to_duration_only_when_no_step_count_is_given(self) -> None:
        """``n_steps=None`` is the only case in which ``duration`` is returned."""
        duration, steps, error = SimEngine._resolve_horizon(None, None, 50.0, 7.0, "run_policy")
        assert (duration, steps, error) == (7.0, None, None)

    def test_the_docstring_states_the_same_rule_on_the_parameter_it_governs(self) -> None:
        """``duration``'s own entry says it is read only when ``n_steps`` is None."""
        doc = inspect.getdoc(PolicyRunner.run) or ""
        assert "Used only when ``n_steps`` is None" in doc


class TestTheStepCountIsRefusedWheneverItIsGiven:
    """Every unusable ``n_steps`` is refused, and the refusal is actionable."""

    @pytest.mark.parametrize("value", UNUSABLE_STEPS, ids=_ident)
    def test_an_unusable_step_count_raises(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _run(n_steps=value)
        text = str(excinfo.value)
        assert "n_steps" in text, text
        assert "PolicyRunner.run" in text, text
        assert repr(value) in text or str(value) in text, text

    @pytest.mark.parametrize("value", UNUSABLE_STEPS, ids=_ident)
    def test_the_refusal_is_not_a_bare_conversion_error(self, value: Any) -> None:
        """The pre-fix failures named a library internal, not the parameter."""
        with pytest.raises(ValueError) as excinfo:
            _run(n_steps=value)
        text = str(excinfo.value)
        for leaked in ("cannot convert float", "not supported between instances", "can't multiply sequence"):
            assert leaked not in text, text


class TestTheDurationIsRefusedWhenItIsTheHorizon:
    """Every unusable ``duration`` is refused when no step count was given."""

    @pytest.mark.parametrize("value", UNUSABLE_DURATION, ids=_ident)
    def test_an_unusable_duration_raises(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _run(duration=value)
        text = str(excinfo.value)
        assert "duration" in text, text
        assert "PolicyRunner.run" in text, text

    @pytest.mark.parametrize("value", [0, 0.0, -5, -0.5], ids=_ident)
    def test_a_zero_length_rollout_is_no_longer_a_completed_budget(self, value: Any) -> None:
        """The worst pre-fix outcome: ``success`` over a rollout that never ran.

        ``steps_used=0`` with ``stopped_reason="budget"`` asserts the horizon was
        exhausted; there was no horizon, and no action was ever applied.
        """
        with pytest.raises(ValueError):
            _run(duration=value)


class TestTheRefusalPrecedesTheRollout:
    """Nothing is queried, applied, or reseeded on a refused horizon."""

    @pytest.mark.parametrize("value", [0, -5, math.nan, "8", [8]], ids=_ident)
    def test_a_refused_step_count_queries_no_policy_and_applies_no_action(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            PolicyRunner(sim).run("arm", policy, control_frequency=50.0, action_horizon=1, n_steps=value)
        assert policy.infer_calls == 0, "the refused rollout queried the policy"
        assert sim.send_count == 0, "the refused rollout applied an action"

    @pytest.mark.parametrize("value", [0, -0.5, math.inf, "10", None], ids=_ident)
    def test_a_refused_duration_queries_no_policy_and_applies_no_action(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            PolicyRunner(sim).run("arm", policy, control_frequency=50.0, action_horizon=1, duration=value)
        assert policy.infer_calls == 0, "the refused rollout queried the policy"
        assert sim.send_count == 0, "the refused rollout applied an action"

    def test_a_refused_horizon_does_not_reseed_the_process_global_rng(self) -> None:
        """``seed=`` reseeds Python / NumPy / torch, so the refusal precedes it."""
        sim, policy = _arm()
        random.seed(4242)
        before = [random.random() for _ in range(3)]
        random.seed(4242)
        with pytest.raises(ValueError):
            PolicyRunner(sim).run("arm", policy, control_frequency=50.0, action_horizon=1, n_steps=0, seed=7)
        after = [random.random() for _ in range(3)]
        assert after == before, "the refused rollout consumed the global RNG"


class TestTheOtherKnobsValueIsNoLongerSelected:
    """The headline: a step count outside the domain selected ``duration``'s.

    ``n_steps=0`` ran ``duration``'s ``10.0`` default at 50 Hz - 500 control
    steps, 500 applied actions and 126 inferences - under ``status="success"``.
    Any attached recorder wrote a 500-frame episode for a caller who asked for
    zero steps.
    """

    @pytest.mark.parametrize("value", [0, -5, math.nan], ids=_ident)
    def test_a_step_count_outside_the_domain_no_longer_runs_the_duration_default(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            PolicyRunner(sim).run("arm", policy, control_frequency=50.0, action_horizon=1, n_steps=value)
        assert sim.send_count == 0, f"n_steps={value!r} still ran {sim.send_count} steps"

    def test_the_duration_default_is_what_it_used_to_run(self) -> None:
        """The premise of the case above: 10.0s at 50 Hz really is 500 steps."""
        assert int(10.0 * 50.0) == 500
        assert inspect.signature(PolicyRunner.run).parameters["duration"].default == 10.0


class TestUsableHorizonsStillRun:
    """Both knobs keep working, and the step accounting is unchanged."""

    @pytest.mark.parametrize("value", USABLE_STEPS, ids=_ident)
    def test_a_usable_step_count_runs_exactly_that_many_steps(self, value: int) -> None:
        result, sim, _policy = _run(n_steps=value)
        assert result["status"] == "success"
        assert _payload(result)["n_steps"] == value
        assert sim.send_count == value

    @pytest.mark.parametrize("value", USABLE_DURATION, ids=_ident)
    def test_a_usable_duration_runs_its_computed_horizon(self, value: float) -> None:
        result, sim, _policy = _run(duration=value)
        expected = int(value * 50.0)
        assert result["status"] == "success"
        assert _payload(result)["n_steps"] == expected
        assert sim.send_count == expected

    def test_the_default_horizon_still_runs(self) -> None:
        """No argument at all keeps running ``duration``'s documented default."""
        result, sim, _policy = _run()
        assert result["status"] == "success"
        assert sim.send_count == 500


class TestDurationIsOnlyJudgedWhenItIsTheHorizon:
    """A value the rollout will not read is not reported on.

    The facade pins this asymmetry for itself -
    ``run_policy(n_steps=2, duration=0)`` succeeds, because ``_resolve_horizon``
    recomputes ``duration`` from the step count - and reporting on ``duration``
    here when a step count was given would make the runner stricter than the
    entry point it serves.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DURATION, ids=_ident)
    def test_an_unusable_duration_is_accepted_alongside_a_usable_step_count(self, value: Any) -> None:
        result, sim, _policy = _run(n_steps=3, duration=value)
        assert result["status"] == "success"
        assert sim.send_count == 3

    def test_a_usable_duration_does_not_override_the_step_count(self) -> None:
        result, sim, _policy = _run(n_steps=3, duration=10.0)
        assert _payload(result)["n_steps"] == 3
        assert sim.send_count == 3


class TestTheDomainIsTheEntryPointsVerbatim:
    """Neither layer may accept what the other refuses, for either knob."""

    @pytest.mark.parametrize("value", [*UNUSABLE_STEPS, *USABLE_STEPS], ids=_ident)
    def test_the_step_count_domain_matches_the_shared_rule(self, value: Any) -> None:
        shared_refuses = positive_count_error(value, "n_steps", "run_policy") is not None
        try:
            _run(n_steps=value)
        except ValueError:
            runner_refuses = True
        else:
            runner_refuses = False
        assert runner_refuses is shared_refuses, f"verdicts differ for n_steps={value!r}"

    @pytest.mark.parametrize("value", [*UNUSABLE_DURATION, *USABLE_DURATION], ids=_ident)
    def test_the_duration_domain_matches_the_shared_rule(self, value: Any) -> None:
        shared_refuses = positive_finite_number_error(value, "duration", "run_policy") is not None
        try:
            _run(duration=value)
        except ValueError:
            runner_refuses = True
        else:
            runner_refuses = False
        assert runner_refuses is shared_refuses, f"verdicts differ for duration={value!r}"

    def test_the_step_count_narrowing_is_the_entry_points_own(self) -> None:
        """A numpy integer is refused here because it is refused there too."""
        assert positive_count_error(np.int64(3), "n_steps", "run_policy") is not None
        _duration, _steps, error = _facade_resolve(np.int64(3))
        assert error is not None

    def test_a_numpy_duration_is_accepted_because_the_shared_rule_accepts_it(self) -> None:
        """``duration``'s domain admits a NumPy scalar, so the runner must too."""
        assert positive_finite_number_error(np.float32(0.1), "duration", "run_policy") is None
        result, sim, _policy = _run(duration=np.float32(0.1))
        assert result["status"] == "success"
        assert sim.send_count == 5


class TestNoRunnerHorizonSurfaceDrifts:
    """Every ``PolicyRunner`` surface taking a horizon knob must judge it.

    A structural sweep rather than a list, so a surface added later cannot
    inherit the gap this module closes.
    """

    @staticmethod
    def _runner_methods(source: str) -> dict[str, ast.FunctionDef]:
        """Public methods of the ``PolicyRunner`` class in ``source``."""
        found: dict[str, ast.FunctionDef] = {}
        for cls in ast.parse(source).body:
            if not isinstance(cls, ast.ClassDef) or cls.name != "PolicyRunner":
                continue
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    found[node.name] = node
        return found

    @staticmethod
    def _guarded(fn: ast.FunctionDef, param: str) -> bool:
        """Does ``fn`` pass ``param`` to a domain helper by name?"""
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if not name.endswith("_error"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == param:
                    return True
        return False

    @staticmethod
    def _horizon_params(fn: ast.FunctionDef) -> list[str]:
        """The horizon knobs ``fn`` declares."""
        declared = {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)}
        return [p for p in ("duration", "n_steps") if p in declared]

    def _source(self) -> str:
        return pathlib.Path(inspect.getfile(PolicyRunner)).read_text(encoding="utf-8")

    def test_the_sweep_finds_the_surface_it_is_written_for(self) -> None:
        """Non-vacuity: ``run`` is found, and it is the only such surface."""
        methods = self._runner_methods(self._source())
        carrying = {n: self._horizon_params(f) for n, f in methods.items() if self._horizon_params(f)}
        assert carrying == {"run": ["duration", "n_steps"]}, carrying

    def test_every_horizon_knob_on_every_surface_is_judged(self) -> None:
        adrift: list[str] = []
        for name, fn in self._runner_methods(self._source()).items():
            for param in self._horizon_params(fn):
                if not self._guarded(fn, param):
                    adrift.append(f"{name}({param})")
        assert adrift == [], f"horizon knob(s) with no domain: {adrift}"

    def test_the_sweep_detects_a_planted_unguarded_surface(self) -> None:
        """A scanner that matched nothing would report a clean sweep."""
        planted = self._source() + (
            "\n\nclass PolicyRunner:\n"
            "    def run_twice(self, robot_name, *, duration=10.0, n_steps=None):\n"
            '        """A surface that reads a horizon and judges neither knob."""\n'
            "        return {}\n"
        )
        adrift = [
            f"{name}({param})"
            for name, fn in self._runner_methods(planted).items()
            for param in self._horizon_params(fn)
            if not self._guarded(fn, param)
        ]
        assert sorted(adrift) == ["run_twice(duration)", "run_twice(n_steps)"], adrift
