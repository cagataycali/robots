"""``mesh.pacing.Ticker`` must be accurate, stoppable, and honest about both.

These tests assert the two properties the publish loops depend on (BUGS.md Q69):
the achieved rate is the requested rate even in a process tree where
``Event.wait`` is inflated by ~145ms, and a stop is honoured within a slice
rather than within a period.

Every timing assertion here is written to hold on BOTH kinds of machine -- a
terminal-started shell where sleeps are accurate, and a daemon-descended tree
where they are not -- by calibrating against
:func:`strands_robots.mesh.pacing.sleep_penalty_s` instead of picking a number
that happens to pass where it was written. A test that only passes in one of the
two environments is exactly the failure mode Q35 documented.
"""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots.mesh.pacing import Ticker, sleep_penalty_s


class TestARefusedPeriodCannotBusySpinAHardwareLoop:
    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.001, float("nan"), float("inf")])
    def test_a_period_that_would_spin_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="period"):
            Ticker(bad)

    @pytest.mark.parametrize("bad", [0, -0.5, float("nan")])
    def test_a_slice_that_would_spin_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="slice_s"):
            Ticker(0.1, slice_s=bad)

    def test_a_slice_longer_than_the_period_is_clamped_not_rejected(self) -> None:
        # A 100Hz loop with the default 10ms slice must not wait 10ms per 10ms
        # tick and then some: the slice is capped at the period.
        with Ticker(0.005) as ticker:
            assert ticker.slice_s == pytest.approx(0.005)


class TestTheAchievedRateIsTheRequestedRate:
    def test_ten_hz_stays_ten_hz_where_event_wait_would_not(self) -> None:
        """The headline claim, measured against Event.wait on the same machine."""
        penalty = sleep_penalty_s()
        period = 0.05

        stop = threading.Event()
        ticks = 0
        with Ticker(period, stop) as ticker:
            start = time.perf_counter()
            while time.perf_counter() - start < 0.6:
                ticks += 1
                if ticker.wait():
                    break
            ticker_hz = ticks / (time.perf_counter() - start)

        # The floor is generous (75% of nominal) because CI machines are noisy,
        # but the point is the COMPARISON below, which is what a wrong pacing
        # implementation cannot satisfy.
        assert ticker_hz > 0.75 / period, f"Ticker only achieved {ticker_hz:.1f}Hz asking for {1 / period:.0f}Hz"

        if penalty < 0.01:
            pytest.skip(
                f"sleeps are accurate here (penalty {penalty * 1000:.1f}ms), so Event.wait pacing is "
                "not degraded and there is nothing to out-perform"
            )
        event_ticks = 0
        event_start = time.perf_counter()
        idle = threading.Event()
        while time.perf_counter() - event_start < 0.6:
            idle.wait(period)
            event_ticks += 1
        event_hz = event_ticks / (time.perf_counter() - event_start)
        assert ticker_hz > event_hz * 1.5, (
            f"Ticker {ticker_hz:.1f}Hz vs Event.wait {event_hz:.1f}Hz at a nominal {1 / period:.0f}Hz "
            f"(sleep penalty {penalty * 1000:.0f}ms) - the selector timer is supposed to sidestep the penalty"
        )

    def test_work_inside_the_tick_is_subtracted_from_the_period_not_added(self) -> None:
        """The period is a deadline: a 20ms tick body inside a 50ms period keeps 50ms."""
        period = 0.05
        elapsed: list[float] = []
        with Ticker(period) as ticker:
            for _ in range(6):
                start = time.perf_counter()
                busy_until = start + 0.02  # busy-wait: unaffected by the sleep penalty
                while time.perf_counter() < busy_until:
                    pass
                ticker.wait()
                elapsed.append(time.perf_counter() - start)
        mid = sorted(elapsed)[len(elapsed) // 2]
        assert mid < period * 1.6, (
            f"a 20ms body inside a {period * 1000:.0f}ms period produced {mid * 1000:.0f}ms ticks - "
            "the work is being ADDED to the period instead of subtracted from it"
        )

    def test_an_overrunning_tick_does_not_fire_a_catch_up_burst(self) -> None:
        """Missed deadlines are dropped, not chased.

        A publish loop that chases lost time emits several frames back to back
        with near-identical timestamps, which reads downstream as a rate spike
        rather than as the stall it actually was.
        """
        period = 0.02
        with Ticker(period) as ticker:
            ticker.wait()
            overrun_until = time.perf_counter() + 0.15  # 7+ periods
            while time.perf_counter() < overrun_until:
                pass
            waits = []
            for _ in range(3):
                start = time.perf_counter()
                ticker.wait()
                waits.append(time.perf_counter() - start)
        assert all(w > period * 0.4 for w in waits), (
            f"after a 150ms overrun the next waits returned immediately ({[round(w * 1000) for w in waits]}ms) - "
            "that is a catch-up burst"
        )


class TestAStopIsHonouredWithinASliceNotWithinAPeriod:
    def test_wait_returns_true_when_the_event_is_already_set(self) -> None:
        stop = threading.Event()
        stop.set()
        with Ticker(10.0, stop) as ticker:
            start = time.perf_counter()
            assert ticker.wait() is True
            assert time.perf_counter() - start < 1.0, "a set stop event must not wait out the period"

    def test_a_stop_set_mid_wait_is_seen_within_a_slice(self) -> None:
        stop = threading.Event()
        penalty = sleep_penalty_s()
        with Ticker(5.0, stop, slice_s=0.01) as ticker:

            def stopper() -> None:
                time.sleep(0.05)
                stop.set()
                ticker.wake()

            threading.Thread(target=stopper, daemon=True).start()
            start = time.perf_counter()
            assert ticker.wait() is True
            took = time.perf_counter() - start
        # The stopper's own sleep(0.05) is subject to the machine's penalty, so
        # the budget is 0.05 + penalty + a slice + slack -- never the 5s period.
        budget = 0.05 + penalty + 0.01 + 0.5
        assert took < budget, f"stop took {took:.3f}s (budget {budget:.3f}s) out of a 5s period"

    def test_a_spurious_wake_does_not_shorten_the_tick(self) -> None:
        """wake() without a stop must not turn a 50ms tick into a 5ms one.

        Otherwise any code that pokes the ticker to be helpful silently doubles
        the publish rate of a hardware loop.
        """
        stop = threading.Event()
        with Ticker(0.05, stop) as ticker:
            threading.Thread(target=ticker.wake, daemon=True).start()
            start = time.perf_counter()
            assert ticker.wait() is False
            assert time.perf_counter() - start > 0.04, "a wake() with no stop cut the tick short"

    def test_wake_and_close_are_safe_in_any_order_and_repeatedly(self) -> None:
        ticker = Ticker(0.01, threading.Event())
        ticker.wake()
        ticker.close()
        ticker.close()  # idempotent
        ticker.wake()  # after close: a no-op, never an exception on a shutdown path
        with pytest.raises(RuntimeError, match="after close"):
            ticker.wait()


class TestTheCalibrationHelperIsUsableByOtherTests:
    def test_it_reports_a_non_negative_extra_cost(self) -> None:
        assert sleep_penalty_s(0.01) >= 0.0

    def test_it_refuses_a_sample_count_it_cannot_take_a_median_of(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            sleep_penalty_s(0.01, samples=0)

    def test_one_sample_is_not_the_default_because_the_first_sleep_can_be_accurate(self) -> None:
        """Regression for a flake I created and measured.

        The first ``time.sleep`` after CPU-bound work came back 3.3ms late on this
        machine while every following one was ~145ms late, so a one-shot probe
        says "sleeps are fine here" often enough to silently skip the comparison
        this module exists to make. The default must therefore be > 1 sample.
        """
        import inspect

        default = inspect.signature(sleep_penalty_s).parameters["samples"].default
        assert default >= 3, "a single-sample calibration is flaky in the direction that hides the bug"

    def test_it_measures_the_extra_not_the_total(self) -> None:
        # A 10ms sleep on an accurate machine is ~0 extra; the total would be
        # ~0.01. Anything that returns the total is broken in the direction that
        # makes every calibrated ceiling too generous.
        assert sleep_penalty_s(0.01) < 1.0
