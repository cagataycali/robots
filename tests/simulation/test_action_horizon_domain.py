"""Accepted domain of ``action_horizon`` on the directly-drivable runner surfaces.

``action_horizon`` is how many actions are consumed from one policy chunk before
re-querying. Every public entry point that accepts it - ``SimEngine.run_policy``
/ ``eval_policy`` / ``evaluate_benchmark`` and ``MuJoCoSimEngine.start_policy`` /
``run_multi_policy`` - refuses a non-positive integer through
:meth:`SimEngine._validate_action_horizon`, whose docstring states the reason:
such a value "would otherwise be silently clamped to 1 by
:func:`~strands_robots.policies.base.resolve_chunk_length`, hiding the caller's
mistake behind a rollout that does not run the requested horizon", and it
delegates the domain to :func:`~strands_robots.utils.positive_count_error` "so a
horizon refused for a simulated rollout cannot be accepted for the real arm".

``PolicyRunner.run`` and ``PolicyRunner.evaluate`` are the layer that consumes
the value, and they are documented as drivable directly. They applied no domain,
so a direct caller got exactly the clamp the entry points exist to prevent:
``0``, ``-5``, ``True``, ``2.7`` and ``"8"`` each ran a rollout to
``status="success"`` at a re-query interval nobody asked for, while ``nan`` /
``inf`` / ``None`` / a list leaked a bare ``int()`` conversion error out of the
FIRST inference - surfaced by ``run`` as "Policy failed: cannot convert float NaN
to integer", naming neither the parameter nor the method, and propagating
uncaught out of ``evaluate``.

Both sibling knobs of the same signature already carry that guarantee:
``control_substeps`` raises from :meth:`PolicyRunner._control_substeps`, whose
docstring calls the raise "the guarantee for callers driving ``PolicyRunner``
directly", and ``control_frequency`` raises from
``policy.set_control_frequency``. ``action_horizon`` was the third.

These tests pin, on the values rather than on wording:

* the premise - what :func:`resolve_chunk_length` really does with each value,
* both runner surfaces refusing every one of them,
* the refusal landing before any inference is queried or action applied,
* usable horizons still running, and the recorded step accounting unchanged,
* the domain being the entry point's verbatim, including its numpy narrowing,
* the check being unconditional rather than gated on the policy carrying RTC
  state (which makes the horizon inert but is not a property of the request),
* and a structural sweep so a future runner surface cannot forget the domain.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import numpy as np
import pytest

import strands_robots.simulation.policy_runner as runner_mod
from strands_robots.policies.base import resolve_chunk_length
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import positive_count_error

from .test_policy_runner_async_rtc import _ChunkPolicy, _CountingSim
from .test_rtc_requery_interval import _CountingSim as _RtcCountingSim
from .test_rtc_requery_interval import _RtcChunkPolicy

# Values no re-query interval can be built from. ``0``/negatives/``False``
# because a chunk cannot yield fewer than one action; ``True`` because ``bool``
# is an ``int`` subclass and would act as a silent horizon of 1; ``2.7`` because
# a fractional horizon truncates to one nobody typed; ``nan``/``inf``/``None``/a
# list because ``int()`` cannot convert them at all; ``"8"`` because a string
# that happens to parse is still not the declared type. The numpy and integral
# float spellings are refused by the entry point's own strict-int domain, so
# including them here is parity rather than a new narrowing - see
# ``TestTheDomainIsTheEntryPointsVerbatim``.
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

# Values a re-query interval CAN be built from: the parameter is a plain
# positive ``int`` and its default is 8.
USABLE: list[int] = [1, 2, 4, 8, 30]


def _ident(value: Any) -> str:
    """A stable label; ``0``/``False`` and ``1``/``True`` hash equal."""
    if value is True:
        return "True"
    if value is False:
        return "False"
    return repr(value)


def _chunk_len(policy: Any, horizon: Any) -> int:
    """``resolve_chunk_length`` with both arguments deliberately un-narrowed.

    The premise tests hand it a duck-typed single-action stand-in rather than a
    ``Policy``, and horizons outside the declared ``int`` - which is the whole
    point of measuring what it does with them. Funnelled through one annotated
    helper so those intentions are stated once instead of suppressed per call.
    """
    return resolve_chunk_length(policy, horizon)


def _arm() -> tuple[_CountingSim, _ChunkPolicy]:
    """A GL-free engine plus a chunk-emitting policy that records its calls."""
    sim = _CountingSim()
    policy = _ChunkPolicy(sim, infer_sleep=0.0)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


class _SingleAction:
    """Duck-typed single-action policy, so the clamp is observable at all.

    A chunk-emitting policy hides it: ``resolve_chunk_length`` returns
    ``max(action_horizon, execution_horizon)``, so any horizon at or below the
    chunk length yields the chunk length and ``0`` is indistinguishable from a
    correct request.
    """

    execution_horizon = 1
    supports_rtc = False


class TestTheConsumerReallyClampsOrRaises:
    """The premise the domain rests on, measured against the real consumer.

    Without this the accepted set would be an assertion about
    ``resolve_chunk_length`` rather than a measurement of it, and a future
    version that started bounding the value itself would leave this domain
    quietly redundant instead of quietly over-strict.
    """

    @pytest.mark.parametrize("value", [0, -5, -1, True, False], ids=_ident)
    def test_a_horizon_below_one_is_silently_clamped_to_one(self, value: Any) -> None:
        assert _chunk_len(_SingleAction(), value) == 1

    def test_a_fractional_horizon_truncates_to_one_nobody_asked_for(self) -> None:
        assert _chunk_len(_SingleAction(), 2.7) == 2

    def test_a_numeric_string_is_accepted_as_a_horizon(self) -> None:
        assert _chunk_len(_SingleAction(), "8") == 8

    @pytest.mark.parametrize(
        ("value", "exc"),
        [(math.nan, ValueError), (math.inf, OverflowError), (None, TypeError), ([4], TypeError)],
        ids=["nan", "inf", "None", "[4]"],
    )
    def test_a_value_int_cannot_convert_raises_from_the_consumer(self, value: Any, exc: type) -> None:
        with pytest.raises(exc):
            _chunk_len(_SingleAction(), value)


class TestBothRunnerSurfacesRefuseTheHorizon:
    """Neither directly-drivable surface may accept a horizon it cannot run."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=_ident)
    def test_run_refuses(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match="action_horizon"):
            PolicyRunner(sim).run("arm", policy, n_steps=8, action_horizon=value)

    @pytest.mark.parametrize("value", UNUSABLE, ids=_ident)
    def test_evaluate_refuses(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError, match="action_horizon"):
            PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=8, action_horizon=value)

    def test_the_message_names_the_surface_the_caller_called(self) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError) as run_exc:
            PolicyRunner(sim).run("arm", policy, n_steps=4, action_horizon=0)
        sim2, policy2 = _arm()
        with pytest.raises(ValueError) as eval_exc:
            PolicyRunner(sim2).evaluate("arm", policy2, n_episodes=1, max_steps=4, action_horizon=0)
        assert "PolicyRunner.run: action_horizon" in str(run_exc.value)
        assert "PolicyRunner.evaluate: action_horizon" in str(eval_exc.value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, None, [4]], ids=["nan", "inf", "None", "[4]"])
    def test_the_bare_conversion_error_no_longer_escapes(self, value: Any) -> None:
        """The refusal replaces the consumer's ``int()`` message, not adds to it."""
        sim, policy = _arm()
        with pytest.raises(ValueError) as exc:
            PolicyRunner(sim).run("arm", policy, n_steps=4, action_horizon=value)
        text = str(exc.value)
        assert "action_horizon" in text
        assert "int()" not in text
        assert "convert" not in text


class TestTheRefusalPrecedesAnyInference:
    """A horizon the runner cannot honor must cost no model call and no command.

    On the pre-fix tree the value was first read inside the chunk query, so the
    policy had already been inferred once - a full forward pass for a real VLA -
    and, when a recorder is attached, the episode was already open.
    """

    @pytest.mark.parametrize("value", [0, 2.7, math.nan, None], ids=_ident)
    def test_run_queries_no_policy_and_sends_no_action(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            PolicyRunner(sim).run("arm", policy, n_steps=8, action_horizon=value)
        assert policy.infer_starts == []
        assert sim.send_count == 0

    @pytest.mark.parametrize("value", [0, 2.7, math.nan, None], ids=_ident)
    def test_evaluate_queries_no_policy_and_sends_no_action(self, value: Any) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError):
            PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=8, action_horizon=value)
        assert policy.infer_starts == []
        assert sim.send_count == 0


class TestUsableHorizonsStillRun:
    """Over-reach control: the guard refuses exactly the unusable values."""

    @pytest.mark.parametrize("value", USABLE)
    def test_run_accepts_a_positive_integer(self, value: int) -> None:
        sim, policy = _arm()
        result = PolicyRunner(sim).run("arm", policy, n_steps=8, action_horizon=value, fast_mode=True)
        assert result["status"] == "success"
        assert sim.send_count == 8

    @pytest.mark.parametrize("value", USABLE)
    def test_evaluate_accepts_a_positive_integer(self, value: int) -> None:
        sim, policy = _arm()
        result = PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=8, action_horizon=value)
        assert result["status"] == "success"
        assert sim.send_count == 8

    def test_the_default_horizon_is_untouched(self) -> None:
        """The parameter's own default must remain inside the domain."""
        default = inspect.signature(PolicyRunner.run).parameters["action_horizon"].default
        assert positive_count_error(default, "action_horizon", "PolicyRunner.run") is None
        assert inspect.signature(PolicyRunner.evaluate).parameters["action_horizon"].default == default


class TestTheDomainIsTheEntryPointsVerbatim:
    """One rule, so a horizon cannot be refused at the facade and run below it."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE], ids=_ident)
    def test_the_runner_refuses_exactly_what_the_entry_point_refuses(self, value: Any) -> None:
        entry_refuses = SimEngine._validate_action_horizon(value, "run_policy") is not None
        sim, policy = _arm()
        try:
            PolicyRunner(sim).run("arm", policy, n_steps=4, action_horizon=value, fast_mode=True)
            runner_refuses = False
        except ValueError:
            runner_refuses = True
        assert runner_refuses is entry_refuses, f"diverged for {value!r}"

    def test_the_wording_is_the_shared_rule_with_this_surface_as_the_context(self) -> None:
        sim, policy = _arm()
        with pytest.raises(ValueError) as exc:
            PolicyRunner(sim).run("arm", policy, n_steps=4, action_horizon=-5)
        assert str(exc.value) == positive_count_error(-5, "action_horizon", "PolicyRunner.run")

    @pytest.mark.parametrize("value", [np.int64(3), np.float64(4.0), 3.0], ids=_ident)
    def test_the_numpy_narrowing_is_the_entry_points_contract_not_a_new_one(self, value: Any) -> None:
        """These are refused above too, so parity - not a narrowing this adds.

        ``positive_count_error`` is the strict-``int`` domain because the value
        is consumed as a slice bound, where an integral float raises rather than
        coercing. Pinned so the shared choice is not read as collateral here.
        """
        assert SimEngine._validate_action_horizon(value, "run_policy") is not None
        sim, policy = _arm()
        with pytest.raises(ValueError, match="action_horizon"):
            PolicyRunner(sim).run("arm", policy, n_steps=4, action_horizon=value)


class TestTheCheckIsUnconditional:
    """Not gated on the policy, matching the entry point.

    ``resolve_chunk_length`` returns early for an RTC policy - it owns its own
    re-query interval - so the horizon is genuinely inert there. Gating the guard
    on that would make the runner accept, for one class of policy, a value its
    own entry point refuses for every class: the divergence this closes, one
    layer narrower.
    """

    @pytest.mark.parametrize("value", [0, -5, 2.7, math.nan], ids=_ident)
    def test_an_rtc_policy_is_refused_the_same_horizon(self, value: Any) -> None:
        sim = _RtcCountingSim()
        policy = _RtcChunkPolicy()
        policy.bind(sim)
        policy.set_robot_state_keys(sim.robot_joint_names("arm"))
        assert policy.supports_rtc is True
        with pytest.raises(ValueError, match="action_horizon"):
            PolicyRunner(sim).run("arm", policy, n_steps=8, action_horizon=value, async_rtc=False)

    def test_the_horizon_really_is_inert_for_an_rtc_policy(self) -> None:
        """The premise the paragraph above rests on, measured."""

        class _Rtc(_SingleAction):
            supports_rtc = True
            execution_horizon = 3

        assert _chunk_len(_Rtc(), 99) == 3
        assert _chunk_len(_Rtc(), 1) == 3


def _surfaces_taking_the_horizon() -> dict[str, list[str]]:
    """Public ``PolicyRunner`` methods that declare ``action_horizon``."""
    source = pathlib.Path(inspect.getfile(runner_mod)).read_text()
    found: dict[str, list[str]] = {}
    for cls in ast.walk(ast.parse(source)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name.startswith("_"):
                continue
            args = method.args.args + method.args.kwonlyargs
            if any(a.arg == "action_horizon" for a in args):
                found.setdefault(cls.name, []).append(method.name)
    return found


def _guards(source: str, func: ast.FunctionDef) -> bool:
    """Whether ``func`` runs the shared domain over its own ``action_horizon``."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if name not in {"positive_count_error", "_validate_action_horizon"}:
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "action_horizon" in segment:
            return True
    return False


class TestEveryRunnerSurfaceOwnsTheDomain:
    """A structural sweep, so a new public runner surface cannot forget it."""

    def test_the_expected_surfaces_are_discovered(self) -> None:
        """Non-vacuity: an empty or mis-rooted sweep must not read as clean."""
        assert _surfaces_taking_the_horizon() == {"PolicyRunner": ["run", "evaluate"]}

    def test_each_public_surface_validates_the_horizon(self) -> None:
        source = pathlib.Path(inspect.getfile(runner_mod)).read_text()
        tree = ast.parse(source)
        adrift: list[str] = []
        for cls_name, methods in _surfaces_taking_the_horizon().items():
            for method_name in methods:
                func = next(
                    m
                    for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == cls_name
                    for m in n.body
                    if isinstance(m, ast.FunctionDef) and m.name == method_name
                )
                if not _guards(source, func):
                    adrift.append(f"{cls_name}.{method_name}")
        assert not adrift, f"accept action_horizon without validating it: {adrift}"

    def test_the_sweep_detects_a_planted_surface(self) -> None:
        """The scanner must fail on an unguarded surface, not merely find none."""
        planted = ast.parse(
            "class PolicyRunner:\n"
            "    def rollout(self, policy, action_horizon=8):\n"
            "        return resolve_chunk_length(policy, action_horizon)\n"
        )
        cls = planted.body[0]
        assert isinstance(cls, ast.ClassDef)
        func = cls.body[0]
        assert isinstance(func, ast.FunctionDef)
        assert not _guards(ast.unparse(planted), func)

    def test_the_private_spec_path_is_reached_only_through_evaluate(self) -> None:
        """Why the sweep may skip it: it is a relay, not an entry point.

        ``_evaluate_with_spec`` consumes ``action_horizon`` but takes it verbatim
        from ``evaluate``, which validates first. Pinned so a second caller
        cannot appear without this failing.
        """
        source = pathlib.Path(inspect.getfile(runner_mod)).read_text()
        tree = ast.parse(source)
        callers = {
            method.name
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef)
            for method in cls.body
            if isinstance(method, ast.FunctionDef)
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_evaluate_with_spec"
        }
        assert callers == {"evaluate"}
