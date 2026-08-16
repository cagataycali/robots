"""A non-finite action stream must not be diagnosed as a near-zero one.

``ZeroActionMonitor`` classifies a per-step ``max(abs(action))`` so the operator
of a VLA that "runs but does not move" is pointed at the right cause. It had two
buckets -- real motion and near-zero -- and forced a third case into the wrong
one:

* ``nan`` compares ``False`` against every threshold, so a poisoned action
  stream advanced the near-zero streak and produced the near-zero warning
  BYTE-IDENTICALLY to a genuinely dead policy, prescribing the obs/rename
  pipeline for a fault that pipeline cannot cause. A single ``nan`` component
  makes ``np.abs(...).max()`` ``nan``, so a vector with five real commands in it
  was reported as emitting no command at all.
* ``inf`` compares ``True``, so it cleared the streak instead: an action every
  backend refuses outright went entirely unreported.

The two constructor knobs shared the root cause. Their bare comparisons
(``threshold < 0``, ``patience < 1``) are ``False`` for ``nan``/``inf`` and let a
``bool`` through as ``1``, so a threshold of ``nan``/``inf``/``True`` made the
watchdog fire on a healthy policy and a patience of ``nan``/``inf`` silently
disabled it.

This pins the classification: which fault each stream is reported as, that the
near-zero contract is unchanged, and the accepted domain of both knobs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies.lerobot_local.embodiment import ZeroActionMonitor
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.utils import finite_number_error, positive_whole_number_error

NAN = float("nan")
INF = float("inf")

# Values no threshold can be honored as. ``-1.0`` is refused by the floor this
# class decides; the rest by the shared numeric rule it delegates to.
UNUSABLE_THRESHOLD: list[Any] = [NAN, INF, -INF, True, False, -1.0, -1e-9, "0.5", None, [0.5], {}]
# ``0`` was explicitly permitted before this change; see the test that pins what it
# actually does (it disables the near-zero report rather than tightening it).
USABLE_THRESHOLD: list[Any] = [0, 0.0, 1e-3, 0.5, 2.0, np.float64(1e-3)]

UNUSABLE_PATIENCE: list[Any] = [0, -5, NAN, INF, True, False, 2.7, "10", None, [10], {}]
USABLE_PATIENCE: list[Any] = [1, 10, 10.0, np.int64(7)]


def _monitor(**kwargs: Any) -> Any:
    """Construct the monitor through one funnel so a deliberately-invalid
    value reaches the runtime guard as a caller would supply it."""
    return ZeroActionMonitor(**kwargs)


def _warnings_for(action: list[float], *, steps: int = 12) -> list[str]:
    """Drive the production wiring and return the warnings a caller would see."""
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3", "4", "5", "6"])
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture()
    logger = logging.getLogger("strands_robots.policies.lerobot_local.policy")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for _ in range(steps):
            policy._tensor_to_action_dicts(torch.tensor(action, dtype=torch.float32))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return captured


class TestANonFiniteStreamIsReportedAsItsOwnFault:
    """The reachable half: ``max_abs_action`` comes straight from model output."""

    def test_a_nan_stream_and_a_dead_policy_no_longer_report_the_same_fault(self):
        """The headline: these two messages used to be byte-identical."""
        dead = _warnings_for([0.0] * 6)
        poisoned = _warnings_for([NAN] * 6)
        assert len(dead) == 1 and len(poisoned) == 1
        assert dead != poisoned
        assert "near-zero actions" in dead[0]
        assert "non-finite action" in poisoned[0]
        assert "near-zero actions" not in poisoned[0]

    def test_the_non_finite_report_does_not_claim_the_magnitude_was_below_the_threshold(self):
        """``nan`` is not ``< 0.001``; it is not comparable."""
        (message,) = _warnings_for([NAN] * 6)
        assert "max abs <" not in message
        assert "max abs = nan" in message

    def test_the_non_finite_report_does_not_prescribe_the_observation_pipeline(self):
        """A missing or mis-renamed observation key produces the near-zero case."""
        (message,) = _warnings_for([NAN] * 6)
        assert "not implicated" in message
        assert "normalization statistics" in message

    def test_a_single_non_finite_component_is_not_reported_as_no_command_at_all(self):
        """Five of the six actuators are commanded, so "near-zero" would be false."""
        (message,) = _warnings_for([0.4, -0.3, NAN, 0.2, 0.1, -0.5])
        assert "non-finite action" in message
        assert "near-zero actions" not in message

    def test_an_infinite_stream_is_reported_rather_than_clearing_the_streak(self):
        """``inf >= threshold`` used to read as real motion, so nothing warned."""
        (message,) = _warnings_for([INF] * 6)
        assert "non-finite action" in message
        assert "max abs = inf" in message

    def test_the_non_finite_report_is_emitted_once_not_per_control_tick(self):
        monitor = _monitor(patience=3)
        fired = [monitor.update(NAN) for _ in range(20)]
        assert len([message for message in fired if message]) == 1

    def test_the_two_faults_are_tracked_independently(self):
        """A stream that is genuinely both still reports both."""
        monitor = _monitor(patience=3)
        fired = [monitor.update(value) for value in (0.0, 0.0, NAN, 0.0, 0.0)]
        reported = [message for message in fired if message]
        assert len(reported) == 2
        assert any("non-finite action" in message for message in reported)
        assert any("near-zero actions" in message for message in reported)

    def test_a_non_finite_step_neither_advances_nor_clears_the_near_zero_streak(self):
        monitor = _monitor(patience=2)
        assert monitor.update(0.0) is None
        assert monitor.update(NAN) is not None  # its own fault, reported
        near_zero = monitor.update(0.0)  # the second genuinely near-zero step
        assert near_zero is not None and "near-zero actions" in near_zero

    def test_reset_re_arms_the_non_finite_report(self):
        monitor = _monitor()
        assert monitor.update(NAN) is not None
        assert monitor.update(NAN) is None
        monitor.reset()
        assert monitor.update(NAN) is not None

    def test_a_non_numeric_magnitude_still_raises_as_it_always_did(self):
        with pytest.raises(TypeError):
            _monitor().update("0.5")  # type: ignore[arg-type]


class TestTheNearZeroContractIsUnchanged:
    """Over-reach controls: the fault this class was written for is untouched."""

    def test_a_dead_policy_still_reports_the_original_near_zero_message(self):
        (message,) = _warnings_for([0.0] * 6)
        assert "Policy emitted near-zero actions (max abs < 0.001) for 10 consecutive steps" in message
        assert "obs_rename / camera keys" in message

    def test_a_healthy_policy_reports_nothing(self):
        assert _warnings_for([0.9] * 6) == []

    def test_a_real_action_still_clears_the_near_zero_streak(self):
        monitor = _monitor(patience=3)
        assert all(monitor.update(0.0) is None for _ in range(2))
        assert monitor.update(1.0) is None
        assert all(monitor.update(0.0) is None for _ in range(2))
        assert monitor.update(0.0) is not None


class TestTheWatchdogKnobsHaveADomain:
    """Direct-API: a value that makes the watchdog useless is refused."""

    @pytest.mark.parametrize("value", UNUSABLE_THRESHOLD, ids=repr)
    def test_an_unusable_threshold_is_refused(self, value):
        with pytest.raises(ValueError) as excinfo:
            _monitor(threshold=value)
        assert "threshold" in str(excinfo.value)
        assert "ZeroActionMonitor" in str(excinfo.value)

    @pytest.mark.parametrize("value", USABLE_THRESHOLD, ids=repr)
    def test_a_usable_threshold_is_accepted_and_normalized(self, value):
        monitor = _monitor(threshold=value)
        assert type(monitor.threshold) is float
        assert monitor.threshold == float(value)

    @pytest.mark.parametrize("value", UNUSABLE_PATIENCE, ids=repr)
    def test_an_unusable_patience_is_refused(self, value):
        with pytest.raises(ValueError) as excinfo:
            _monitor(patience=value)
        assert "patience" in str(excinfo.value)
        assert "ZeroActionMonitor" in str(excinfo.value)

    @pytest.mark.parametrize("value", USABLE_PATIENCE, ids=repr)
    def test_a_usable_patience_is_accepted_and_normalized_to_a_step_count(self, value):
        """``range(monitor.patience)`` is how a caller consumes it."""
        monitor = _monitor(patience=value)
        assert type(monitor.patience) is int
        assert len(range(monitor.patience)) == int(value)

    def test_a_zero_threshold_is_accepted_and_disables_the_near_zero_report(self):
        """Zero was explicitly permitted before this change and still is.

        Its measured meaning is not "only an exactly-zero action counts": the
        comparison is ``magnitude >= threshold``, so ``0`` accepts EVERY
        magnitude as motion and the near-zero report never fires. Whether a
        silent disable with no documented spelling should instead be refused is
        a separate question from the classification this module pins, so the
        accepted domain is unchanged in that direction and the behaviour is
        recorded here rather than altered.
        """
        monitor = _monitor(threshold=0.0, patience=2)
        assert all(monitor.update(0.0) is None for _ in range(50))
        # the non-finite report is independent of the threshold, so it survives
        assert monitor.update(NAN) is not None

    def test_the_refusal_names_the_value_it_received(self):
        with pytest.raises(ValueError, match="got nan"):
            _monitor(threshold=NAN)
        with pytest.raises(ValueError, match="got 2.7"):
            _monitor(patience=2.7)

    def test_a_refused_threshold_that_fires_on_a_healthy_policy_is_the_reason(self):
        """Each refused threshold would have warned about a moving arm."""
        for value in (NAN, INF, True):
            with pytest.raises(ValueError):
                _monitor(threshold=value)
        # the accepted default does not
        monitor = _monitor()
        assert all(monitor.update(0.9) is None for _ in range(30))

    def test_a_refused_patience_that_disables_the_warning_is_the_reason(self):
        """Each refused patience would have silenced the warning entirely."""
        for value in (NAN, INF):
            with pytest.raises(ValueError):
                _monitor(patience=value)
        monitor = _monitor(patience=2)
        assert monitor.update(0.0) is None
        assert monitor.update(0.0) is not None


class TestTheFloorIsTheWholeLocalContribution:
    """The shared numeric rule decides everything but the floor."""

    @pytest.mark.parametrize("value", UNUSABLE_THRESHOLD + USABLE_THRESHOLD, ids=repr)
    def test_threshold_delegates_every_decision_but_the_sign(self, value):
        shared_refuses = finite_number_error(value, "threshold", "ZeroActionMonitor") is not None
        below_floor = not shared_refuses and float(value) < 0.0
        try:
            _monitor(threshold=value)
            refused = False
        except ValueError:
            refused = True
        assert refused == (shared_refuses or below_floor)

    @pytest.mark.parametrize("value", UNUSABLE_PATIENCE + USABLE_PATIENCE, ids=repr)
    def test_patience_delegates_every_decision(self, value):
        shared_refuses = positive_whole_number_error(value, "patience", "ZeroActionMonitor") is not None
        try:
            _monitor(patience=value)
            refused = False
        except ValueError:
            refused = True
        assert refused == shared_refuses

    @pytest.mark.parametrize("value", [NAN, INF, True], ids=repr)
    def test_the_premise_the_bare_comparisons_missed(self, value):
        """Why a comparison cannot express either domain.

        The pre-fix guards were ``threshold < 0`` and ``patience < 1``, and
        both are ``False`` for every value here, so every one of them was
        stored: ``nan`` and ``inf`` because any comparison against them is
        ``False``, and ``True`` because it reaches a numeric comparison as the
        ``int`` ``1``.

        The value under test is a parameter rather than a literal. A
        comparison written between two literals is decided when it is typed,
        so it would state this premise without measuring it.
        """
        assert not (value < 0.0)
        assert not (value < 1)

    def test_what_replaced_the_comparisons_refuses_all_three(self):
        """Finiteness sees ``nan``/``inf``; only an explicit test sees ``bool``."""
        assert not math.isfinite(NAN) and not math.isfinite(INF)
        # A ``bool`` is finite, so a finiteness test alone does not see it. The
        # shared rules reject it explicitly, which is why they cover all three
        # where neither a comparison nor finiteness alone does.
        assert isinstance(True, int) and math.isfinite(True)
        for value in (NAN, INF, True):
            assert finite_number_error(value, "threshold", "ZeroActionMonitor") is not None
            assert positive_whole_number_error(value, "patience", "ZeroActionMonitor") is not None
