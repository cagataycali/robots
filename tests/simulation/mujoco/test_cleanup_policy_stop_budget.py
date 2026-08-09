"""``cleanup``'s per-future stop budget is held to a domain the join can measure.

``MuJoCoSimEngine.cleanup`` waits on each live policy Future before nulling the
world, because a worker still inside ``mj_step`` on freed arrays is a
stale-pointer segfault (GH #116). That wait is
``Future.result(timeout=policy_stop_timeout)``, and ``Future.result`` measures
its budget as ``time.monotonic() + timeout``: ``0``, a negative value and
``nan`` expire it before the first check, ``inf`` raises ``OverflowError`` out
of the arithmetic, and a non-real raises ``TypeError``. Every one of them
abandons the worker the join exists to await, and a non-real additionally makes
the ``%.1f`` in the join's own warning raise so the record reporting the skipped
wait is dropped.

These pin that such a budget is reported and resolved to the documented default
rather than used, and - the half a refusal would break - that teardown still
completes. ``None`` remains the "no preference" spelling, and the resolver
returns a plain ``float`` because ``Future.result`` refuses a ``np.float32``
the shared guard accepts.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import (
    MuJoCoSimEngine,
    _resolve_policy_stop_timeout,
)
from strands_robots.utils import positive_finite_number_error

DEFAULT = MuJoCoSimEngine._DEFAULT_POLICY_STOP_TIMEOUT

# Budgets ``Future.result`` cannot measure. Each is grouped by what it does to
# the wait, because the groups need different assertions below.
EXPIRES_IMMEDIATELY: list[Any] = [0, 0.0, -1.0, -0.5, math.nan, -math.inf, False]
RAISES_OVERFLOW: list[Any] = [math.inf]
RAISES_TYPE_ERROR: list[Any] = ["5", [5], {}, object()]
SILENT_ONE_SECOND: list[Any] = [True]
UNUSABLE: list[Any] = [*EXPIRES_IMMEDIATELY, *RAISES_OVERFLOW, *RAISES_TYPE_ERROR, *SILENT_ONE_SECOND]

# Budgets the join can measure, including the NumPy spellings the shared guard
# accepts and a real config array would produce.
USABLE: list[Any] = [0.001, 0.5, 2.0, 5.0, 30.0, np.float32(0.25), np.float64(1.5), np.int64(3)]


class _RecordingFuture:
    """Stand-in for a live policy Future that records the budget it is joined with.

    ``done()`` is ``False`` so ``_prune_done_futures`` keeps it, and ``result``
    returns rather than blocking - the assertion is the budget the join
    *requested*, which is exact on any host, not a wall-clock duration.
    """

    def __init__(self) -> None:
        self.timeouts: list[Any] = []

    def done(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False

    def result(self, timeout: Any = None) -> None:
        self.timeouts.append(timeout)


def _sim() -> Any:
    """A world with no robots - enough for ``cleanup`` to run its join."""
    sim: Any = MuJoCoSimEngine(tool_name="cleanup_budget", mesh=False)
    sim.create_world()
    return sim


def _joined_with(budget: Any, *, supplied: bool = True) -> tuple[list[Any], Any]:
    """Budgets the join requested, plus the sim, after one ``cleanup``."""
    sim = _sim()
    fut = _RecordingFuture()
    sim._policy_threads["arm"] = fut
    if supplied:
        sim.cleanup(policy_stop_timeout=budget)
    else:
        sim.cleanup()
    return fut.timeouts, sim


class TestTheResolverAcceptsExactlyWhatTheSharedRuleDoes:
    """The domain is the shared positive-finite rule, not a second opinion."""

    @pytest.mark.parametrize("budget", UNUSABLE)
    def test_a_budget_the_shared_rule_refuses_resolves_to_the_default(self, budget: Any) -> None:
        assert positive_finite_number_error(budget, "policy_stop_timeout", "cleanup") is not None
        assert _resolve_policy_stop_timeout(budget, DEFAULT) == DEFAULT

    @pytest.mark.parametrize("budget", USABLE)
    def test_a_budget_the_shared_rule_accepts_is_used_as_given(self, budget: Any) -> None:
        assert positive_finite_number_error(budget, "policy_stop_timeout", "cleanup") is None
        assert _resolve_policy_stop_timeout(budget, DEFAULT) == pytest.approx(float(budget))

    def test_none_is_the_documented_no_preference_spelling(self) -> None:
        assert _resolve_policy_stop_timeout(None, DEFAULT) == DEFAULT

    @pytest.mark.parametrize("budget", [*USABLE, *UNUSABLE, None])
    def test_the_resolved_budget_is_always_a_plain_positive_finite_float(self, budget: Any) -> None:
        resolved = _resolve_policy_stop_timeout(budget, DEFAULT)
        assert type(resolved) is float
        assert resolved > 0.0 and math.isfinite(resolved)


class TestAnUnmeasurableBudgetIsReported:
    """A budget that is not used must say so, naming the parameter and the value."""

    @pytest.mark.parametrize("budget", UNUSABLE)
    def test_the_warning_names_the_parameter_and_the_default_it_fell_back_to(
        self, budget: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.simulation"):
            _resolve_policy_stop_timeout(budget, DEFAULT)
        assert "policy_stop_timeout" in caplog.text
        assert f"{DEFAULT:.1f}s" in caplog.text

    @pytest.mark.parametrize("budget", [*USABLE, None])
    def test_a_usable_budget_is_not_reported(self, budget: Any, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.simulation"):
            _resolve_policy_stop_timeout(budget, DEFAULT)
        assert "policy_stop_timeout" not in caplog.text


class TestTheJoinIsAwaitedWithTheResolvedBudget:
    """End to end: what ``cleanup`` hands ``Future.result`` for a live worker."""

    @pytest.mark.parametrize("budget", UNUSABLE)
    def test_an_unmeasurable_budget_does_not_reach_the_join(self, budget: Any) -> None:
        joined, _sim_ = _joined_with(budget)
        assert joined == [DEFAULT]

    @pytest.mark.parametrize("budget", [0.001, 0.5, 2.0, np.float64(1.5)])
    def test_a_usable_budget_reaches_the_join_unchanged(self, budget: Any) -> None:
        joined, _sim_ = _joined_with(budget)
        assert joined == [pytest.approx(float(budget))]

    def test_omitting_the_budget_joins_with_the_documented_default(self) -> None:
        joined, _sim_ = _joined_with(None, supplied=False)
        assert joined == [DEFAULT]

    def test_the_join_is_reached_rather_than_skipped(self) -> None:
        """The worker is still awaited - the budget is resolved, not dropped."""
        joined, _sim_ = _joined_with(math.inf)
        assert len(joined) == 1


class TestTeardownStillCompletesAfterAnUnmeasurableBudget:
    """The half a refusal would break: ``cleanup`` releases everything regardless.

    ``cleanup`` is the release path - ``__exit__`` and the finalizer both call
    it - so raising on a bad budget would leak the world, the executor and the
    renderers for a value error. Resolving to the safe default keeps the
    contract its docstring states.
    """

    @pytest.mark.parametrize("budget", [math.inf, math.nan, 0.0, -1.0, "5", True])
    def test_the_world_and_the_executor_are_released(self, budget: Any) -> None:
        joined, sim = _joined_with(budget)
        assert sim._world is None
        with pytest.raises(RuntimeError):
            sim._executor.submit(lambda: None)

    @pytest.mark.parametrize("budget", [math.inf, "5"])
    def test_cleanup_does_not_raise(self, budget: Any) -> None:
        sim = _sim()
        sim.cleanup(policy_stop_timeout=budget)  # must not raise


class TestTheJoinWarningCanNoLongerBeDropped:
    """A non-real budget used to silence the very record that reported it.

    The join logs ``"...did not stop within %.1fs..."`` with the budget. Python's
    logging calls ``handleError`` when a record cannot be formatted, so with a
    ``str`` budget the operator got neither the wait nor the warning about it.
    The resolver hands that ``%.1f`` a ``float``, so the record always renders.
    """

    @pytest.mark.parametrize("budget", UNUSABLE)
    def test_the_resolved_budget_always_renders_in_the_join_warning(self, budget: Any) -> None:
        resolved = _resolve_policy_stop_timeout(budget, DEFAULT)
        record = logging.LogRecord(
            name="strands_robots.simulation.mujoco.simulation",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="cleanup: policy on '%s' did not stop within %.1fs: %s",
            args=("arm", resolved, "reason"),
            exc_info=None,
        )
        assert f"{DEFAULT:.1f}s" in record.getMessage()

    @pytest.mark.parametrize("budget", ["5", [5]])
    def test_the_raw_budget_would_have_broken_that_same_format(self, budget: Any) -> None:
        record = logging.LogRecord(
            name="strands_robots.simulation.mujoco.simulation",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="cleanup: policy on '%s' did not stop within %.1fs: %s",
            args=("arm", budget, "reason"),
            exc_info=None,
        )
        with pytest.raises(TypeError, match="must be real number"):
            record.getMessage()


class TestNormalizingToFloatIsLoadBearing:
    """``Future.result`` is stricter than the shared guard, so the cast is not cosmetic."""

    def test_future_result_refuses_a_numpy_float32_the_guard_accepts(self) -> None:
        # Bound through ``Any`` because the point is what the runtime does with a
        # budget the annotation does not describe but the shared guard accepts.
        budget: Any = np.float32(0.05)
        assert positive_finite_number_error(budget, "policy_stop_timeout", "cleanup") is None
        executor = ThreadPoolExecutor(max_workers=1)
        gate = threading.Event()
        future = executor.submit(gate.wait, 5.0)
        try:
            with pytest.raises(TypeError):
                future.result(timeout=budget)
            # The resolved value is the same budget in a shape the join accepts.
            with pytest.raises(TimeoutError):
                future.result(timeout=_resolve_policy_stop_timeout(budget, DEFAULT))
        finally:
            gate.set()
            executor.shutdown(wait=True)

    def test_the_join_receives_a_plain_float_for_a_numpy_budget(self) -> None:
        joined, _sim_ = _joined_with(np.float32(0.25))
        assert type(joined[0]) is float
