"""``max_onframe_failures`` must be a limit the failure watchdog can count against.

``on_frame`` is where a backend attaches dataset recording and video capture, so
the runner counts consecutive hook exceptions and aborts the episode after
``max_onframe_failures`` of them - the abort text reads "aborting episode to
avoid silent dataset corruption" (GH #117).

The limit was read straight into ``consecutive_onframe_failures >= limit`` with
no domain, so a value outside it did not resize the tolerance, it silenced the
mechanism. Measured on a 100-step rollout whose hook raises on every step:

    limit      status     hook calls   aborted   warnings emitted
    3          error      3            yes       3
    None (5)   error      5            yes       5
    nan        success    100          NO        0   <- 100 frames lost
    inf        success    100          NO        0   <- 100 frames lost
    0          error      1            yes       "failed 0 times in a row"
    2.7        error      3            yes       "failed 2.7 times in a row"
    '5'        error      1            NO        bare TypeError from '>='

``nan``/``inf`` lose the abort *and* the warning: the per-failure warning
interpolates the limit with ``%d``, and ``"%d" % nan`` raises ``ValueError``
while ``"%d" % inf`` raises ``OverflowError``, so ``logging`` emits its own
error and the operator is told nothing at all.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import pathlib
import re
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation import base as base_mod
from strands_robots.simulation import policy_runner as runner_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import positive_count_error

# Values no integer failure counter can be compared against. ``None`` is
# excluded deliberately: it is this parameter's documented "use the runner's own
# limit" spelling and its default.
UNUSABLE: list[Any] = [0, -5, True, False, 2.7, math.nan, math.inf, -math.inf, "5", [5], {}]
USABLE: list[Any] = [1, 3, 5, 100]


@pytest.fixture
def sim():
    s = Simulation(tool_name="onframe_domain", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _policy(sim):
    p = MockPolicy()
    p.set_robot_state_keys(sim.robot_joint_names("alice"))
    return p


class _AlwaysFails:
    """An ``on_frame`` hook that raises every call and records its call count."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, step: int, observation: dict, action: dict) -> None:
        self.calls += 1
        raise ValueError(f"boom-{step}")


def _text(result: dict) -> str:
    return result["content"][0]["text"] if result.get("content") else ""


class TestTheFacadeRefusesALimitTheWatchdogCannotCountAgainst:
    """``SimEngine.run_policy`` reports the refusal through its documented envelope."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_limit_is_refused(self, sim, value):
        result = sim.run_policy("alice", policy_provider="mock", n_steps=2, fast_mode=True, max_onframe_failures=value)
        assert result["status"] == "error", result
        assert "max_onframe_failures" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_names_the_parameter_the_domain_and_the_value(self, sim, value):
        result = sim.run_policy("alice", policy_provider="mock", n_steps=2, fast_mode=True, max_onframe_failures=value)
        text = _text(result)
        assert "run_policy: max_onframe_failures must be a positive integer" in text, text
        assert repr(value) in text, text

    def test_the_refusal_costs_no_inference(self, sim):
        # Guard placement: a refused limit must not query the policy, so it costs
        # no weight download, no recorded frame and no step. (The hook half is
        # asserted against PolicyRunner, which is the layer that takes on_frame.)
        policy = _policy(sim)
        queried = {"n": 0}
        real = policy.get_actions

        async def counting(*args: Any, **kwargs: Any) -> Any:
            queried["n"] += 1
            return await real(*args, **kwargs)

        policy.get_actions = counting  # type: ignore[method-assign]
        result = sim.run_policy(
            "alice", policy_object=policy, n_steps=50, fast_mode=True, max_onframe_failures=math.nan
        )
        assert result["status"] == "error", result
        assert queried["n"] == 0, "a refused limit must not query the policy"


class TestTheRunnerRefusesTheSameDomain:
    """``PolicyRunner`` is drivable directly, so it raises rather than returning."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_limit_raises(self, sim, value):
        policy = _policy(sim)
        with pytest.raises(ValueError, match="max_onframe_failures must be a positive integer"):
            PolicyRunner(sim).run(
                "alice", policy, duration=0.1, control_frequency=50, fast_mode=True, max_onframe_failures=value
            )

    def test_the_raise_names_the_layer_the_caller_called(self, sim):
        policy = _policy(sim)
        with pytest.raises(ValueError) as excinfo:
            PolicyRunner(sim).run(
                "alice", policy, duration=0.1, control_frequency=50, fast_mode=True, max_onframe_failures=math.inf
            )
        assert "PolicyRunner.run:" in str(excinfo.value), str(excinfo.value)

    def test_the_raise_precedes_the_first_hook_call(self, sim):
        hook = _AlwaysFails()
        policy = _policy(sim)
        with pytest.raises(ValueError):
            PolicyRunner(sim).run(
                "alice",
                policy,
                duration=2.0,
                control_frequency=50,
                fast_mode=True,
                on_frame=hook,
                max_onframe_failures=math.nan,
            )
        assert hook.calls == 0


class TestUsableLimitsStillWork:
    """The change is additive: every limit the counter can honor is unaffected."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_limit_is_accepted(self, sim, value):
        result = sim.run_policy("alice", policy_provider="mock", n_steps=2, fast_mode=True, max_onframe_failures=value)
        assert result["status"] == "success", result

    def test_none_is_accepted_and_uses_the_runners_own_limit(self, sim):
        hook = _AlwaysFails()
        policy = _policy(sim)
        result = PolicyRunner(sim).run(
            "alice",
            policy,
            duration=5.0,
            control_frequency=50,
            fast_mode=True,
            on_frame=hook,
            max_onframe_failures=None,
        )
        assert result["status"] == "error", result
        assert f"failed {runner_mod._MAX_CONSECUTIVE_ONFRAME_FAILURES} times in a row" in _text(result)
        assert hook.calls == runner_mod._MAX_CONSECUTIVE_ONFRAME_FAILURES

    @pytest.mark.parametrize("limit", [1, 3])
    def test_a_usable_limit_aborts_after_exactly_that_many_failures(self, sim, limit):
        hook = _AlwaysFails()
        policy = _policy(sim)
        result = PolicyRunner(sim).run(
            "alice",
            policy,
            duration=5.0,
            control_frequency=50,
            fast_mode=True,
            on_frame=hook,
            max_onframe_failures=limit,
        )
        assert result["status"] == "error", result
        # The abort message quotes the real number of consecutive failures.
        assert f"failed {limit} times in a row" in _text(result)
        assert hook.calls == limit


class TestZeroIsADuplicateSpellingOfOne:
    """Refusing ``0`` costs no capability - it aborted where ``1`` aborts.

    The counter is incremented before the comparison, so a limit of ``1``
    already aborts on the first failure. ``0`` aborted on that same failure and
    reported "failed 0 times in a row" when one failure had occurred, so it was
    a duplicate of ``1`` carrying a false count rather than a distinct
    tolerance.
    """

    def test_one_already_aborts_on_the_first_failure_with_a_true_count(self, sim):
        hook = _AlwaysFails()
        policy = _policy(sim)
        result = PolicyRunner(sim).run(
            "alice", policy, duration=5.0, control_frequency=50, fast_mode=True, on_frame=hook, max_onframe_failures=1
        )
        assert hook.calls == 1, "limit=1 aborts on the first failure"
        assert "failed 1 times in a row" in _text(result)

    def test_zero_is_refused_rather_than_reporting_a_count_of_zero(self, sim):
        policy = _policy(sim)
        with pytest.raises(ValueError, match="max_onframe_failures must be a positive integer"):
            PolicyRunner(sim).run(
                "alice", policy, duration=0.1, control_frequency=50, fast_mode=True, max_onframe_failures=0
            )


class TestWhyANonFiniteLimitSilencedTheWatchdogEntirely:
    """Executable premise for the failure the domain removes.

    Both halves of the mechanism read the limit, and both are lost for a
    non-finite value: the abort compares against it, and the warning
    interpolates it with ``%d``.
    """

    @pytest.mark.parametrize("limit", [math.nan, math.inf])
    @pytest.mark.parametrize("consecutive", [1, 5, 100, 10**9])
    def test_the_abort_comparison_is_false_for_every_counter_value(self, limit, consecutive):
        assert not (consecutive >= limit), "a counter can never reach this limit"

    @pytest.mark.parametrize(("limit", "expected"), [(math.nan, ValueError), (math.inf, OverflowError)])
    def test_the_warning_cannot_be_formatted_against_a_non_finite_limit(self, limit, expected):
        # Exactly the operation logging performs on the warning: a LogRecord
        # interpolates ``msg % args``, and the limit is interpolated with %d.
        record = logging.LogRecord(
            name="probe",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="on_frame hook failed (%d/%d consecutive): %s",
            args=(1, limit, "boom"),
            exc_info=None,
        )
        with pytest.raises(expected):
            record.getMessage()

    def test_logging_drops_the_warning_rather_than_emitting_it(self):
        # So a non-finite limit lost the abort AND the only report of the hook.
        # propagate=False so the record never reaches pytest's own capture
        # handler, which re-raises a format failure instead of swallowing it.
        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    records.append(record.getMessage())
                except (ValueError, OverflowError):
                    pass  # exactly what logging reports as "--- Logging error ---"

        handler = _Collect()
        logger = logging.getLogger("strands_robots.test_onframe_premise")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            logger.warning("on_frame hook failed (%d/%d consecutive): %s", 1, math.nan, "boom")
        finally:
            logger.removeHandler(handler)
            logger.propagate = True
        assert records == [], "the warning is never emitted for a non-finite limit"

    def test_a_usable_limit_does_emit_the_warning(self):
        # Non-vacuity for the assertion above: the same call with a usable limit
        # produces the warning, so its absence is the format failure and not a
        # property of the harness.
        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _Collect()
        logger = logging.getLogger("strands_robots.test_onframe_premise_ok")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            logger.warning("on_frame hook failed (%d/%d consecutive): %s", 1, 5, "boom")
        finally:
            logger.removeHandler(handler)
            logger.propagate = True
        assert records == ["on_frame hook failed (1/5 consecutive): boom"]


class TestTheTwoSurfacesAgree:
    """Neither layer may accept what the other refuses."""

    @pytest.mark.parametrize("value", [*UNUSABLE, *USABLE, None])
    def test_the_facade_and_the_runner_reach_the_same_verdict(self, sim, value):
        facade = sim.run_policy("alice", policy_provider="mock", n_steps=2, fast_mode=True, max_onframe_failures=value)
        facade_refused = facade["status"] == "error" and "max_onframe_failures" in _text(facade)

        policy = _policy(sim)
        try:
            PolicyRunner(sim).run(
                "alice", policy, duration=0.05, control_frequency=50, fast_mode=True, max_onframe_failures=value
            )
            runner_refused = False
        except ValueError as e:
            runner_refused = "max_onframe_failures" in str(e)

        assert facade_refused == runner_refused, f"verdicts differ for max_onframe_failures={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_both_report_the_shared_domain_message(self, sim, value):
        assert positive_count_error(value, "max_onframe_failures", "run_policy") is not None
        assert _text(
            sim.run_policy("alice", policy_provider="mock", n_steps=2, fast_mode=True, max_onframe_failures=value)
        ) == positive_count_error(value, "max_onframe_failures", "run_policy")


# --------------------------------------------------------------------------- #
# Structural guard: a new surface taking the parameter cannot skip the domain.
# --------------------------------------------------------------------------- #

_PARAM = "max_onframe_failures"
_DOMAIN_CALLS = {"_validate_onframe_failure_limit", "positive_count_error"}


def _scan_root() -> pathlib.Path:
    # Derived from a symbol rather than a path literal, so a scan rooted
    # elsewhere fails the non-vacuity assertion below instead of passing.
    return pathlib.Path(inspect.getfile(SimEngine)).parent


def _calls_a_domain_helper(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if name in _DOMAIN_CALLS:
                return True
    return False


def _forwards_the_parameter(fn: ast.AST) -> bool:
    """A wrapper that hands the value on by keyword inherits the callee's domain."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == _PARAM and isinstance(kw.value, ast.Name) and kw.value.id == _PARAM:
                    return True
    return False


def _public_surfaces() -> list[tuple[str, str, ast.AST]]:
    found: list[tuple[str, str, ast.AST]] = []
    for path in sorted(_scan_root().rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - the package parses
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if cls.name.startswith("_"):
                continue
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
                if fn.name.startswith("_"):
                    continue
                args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                if _PARAM in args:
                    found.append((f"{path.name}::{cls.name}.{fn.name}", path.name, fn))
    return found


class TestNoOnframeFailureLimitSurfaceDrifts:
    """Every public method taking the parameter validates or forwards it."""

    def test_the_known_surfaces_are_the_ones_found(self):
        # Non-vacuity: if the scan resolves nothing (or something else), the
        # negative assertion below could pass by matching nothing.
        names = {name for name, _mod, _fn in _public_surfaces()}
        assert names == {
            "base.py::SimEngine.run_policy",
            "policy_runner.py::PolicyRunner.run",
            "simulation.py::MuJoCoSimEngine.run_policy",
        }, names

    def test_every_public_surface_validates_or_forwards(self):
        adrift = [
            name
            for name, _mod, fn in _public_surfaces()
            if not (_calls_a_domain_helper(fn) or _forwards_the_parameter(fn))
        ]
        assert adrift == [], f"{adrift} read {_PARAM} without a domain and without forwarding it"

    def test_the_scanner_detects_a_planted_surface(self):
        planted = ast.parse(
            "class Engine:\n    def run_policy(self, max_onframe_failures=None):\n        return max_onframe_failures\n"
        )
        fn = planted.body[0].body[0]  # type: ignore[attr-defined]
        assert not _calls_a_domain_helper(fn)
        assert not _forwards_the_parameter(fn)

    def test_the_scanner_accepts_a_planted_forwarder(self):
        planted = ast.parse(
            "class Engine:\n"
            "    def run_policy(self, max_onframe_failures=None):\n"
            "        return super().run_policy(max_onframe_failures=max_onframe_failures)\n"
        )
        fn = planted.body[0].body[0]  # type: ignore[attr-defined]
        assert _forwards_the_parameter(fn)


class TestTheDomainIsDiscoverable:
    """A refused value must name a parameter whose documented domain says so."""

    @pytest.mark.parametrize(
        ("method", "owner"),
        [(SimEngine.run_policy, "run_policy"), (PolicyRunner.run, "PolicyRunner.run")],
    )
    def test_the_args_entry_states_the_domain(self, method, owner):
        doc = inspect.getdoc(method) or ""
        entry_start = doc.find(f"\n    {_PARAM}:")
        assert entry_start != -1, f"{owner} has no Args: entry for {_PARAM}"
        # Bound the slice at the next entry so the claim is "in this entry".
        rest = doc[entry_start + 1 :]
        following = re.search(r"\n    (?=\S)", rest[1:])
        end = 1 + following.start() if following else len(rest)
        entry = " ".join(rest[:end].split())
        assert "positive integer" in entry, entry

    def test_the_validator_records_why_a_non_finite_limit_is_refused(self):
        doc = inspect.getdoc(SimEngine._validate_onframe_failure_limit) or ""
        collapsed = " ".join(doc.split())
        assert "the abort never fires" in collapsed, collapsed
        assert "duplicate spelling of ``1``" in collapsed, collapsed

    def test_the_module_docstring_convention_is_unchanged(self):
        # base.py owns the envelope-returning guards; policy_runner.py raises.
        assert "_validate_onframe_failure_limit" in pathlib.Path(inspect.getfile(base_mod)).read_text()
        assert "positive_count_error" in pathlib.Path(inspect.getfile(runner_mod)).read_text()
