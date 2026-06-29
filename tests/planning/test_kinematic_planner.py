"""Behaviour tests for KinematicPlanner: clamping, debounce, threaded polling."""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots.planning import KinematicPlanner, PlannerCommand, PlannerUpdate
from strands_robots.planning.inputs import InputSource


class _ManualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_velocity_and_height_clamped_to_limits() -> None:
    kp = KinematicPlanner(max_speed=1.0, max_omega=2.0, height_range=(0.4, 0.8))
    kp.apply_update(PlannerUpdate(root_vel=(5.0, -5.0, 9.0), height=3.0))
    cmd = kp.poll()
    assert cmd.root_vel == (1.0, -1.0, 2.0)
    assert cmd.height == 0.8


def test_style_change_debounced_within_window() -> None:
    clock = _ManualClock()
    kp = KinematicPlanner(style_debounce_s=0.2, clock=clock)
    kp.apply_update(PlannerUpdate(style="happy"))
    assert kp.poll().style == "happy"
    # Second change inside the debounce window is ignored.
    clock.t = 0.1
    kp.apply_update(PlannerUpdate(style="stealth"))
    assert kp.poll().style == "happy"
    # After the window it is accepted.
    clock.t = 0.4
    kp.apply_update(PlannerUpdate(style="stealth"))
    assert kp.poll().style == "stealth"


def test_stop_request_zeroes_velocity_and_sets_flag() -> None:
    kp = KinematicPlanner(initial=PlannerCommand(root_vel=(0.5, 0.0, 0.0)))
    assert not kp.stop_requested
    kp.apply_update(PlannerUpdate(stop=True))
    assert kp.stop_requested
    assert kp.poll().root_vel == (0.0, 0.0, 0.0)


def test_reset_restores_initial_and_clears_stop() -> None:
    initial = PlannerCommand(root_vel=(0.2, 0.0, 0.0), style="run")
    kp = KinematicPlanner(initial=initial)
    kp.apply_update(PlannerUpdate(root_vel=(0.9, 0.0, 0.0), stop=True))
    kp.reset()
    assert kp.poll() == initial
    assert not kp.stop_requested


def test_empty_update_is_noop() -> None:
    kp = KinematicPlanner(initial=PlannerCommand(root_vel=(0.3, 0.0, 0.0)))
    kp.apply_update(PlannerUpdate())
    assert kp.poll().root_vel == (0.3, 0.0, 0.0)


def test_invalid_construction_args_rejected() -> None:
    with pytest.raises(ValueError, match="max_speed"):
        KinematicPlanner(max_speed=0.0)
    with pytest.raises(ValueError, match="height_range"):
        KinematicPlanner(height_range=(0.8, 0.4))
    with pytest.raises(ValueError, match="style_debounce_s"):
        KinematicPlanner(style_debounce_s=-1.0)


class _BlockingInput(InputSource):
    """Input whose poll() blocks until released - proves polling is off-thread."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.polled = threading.Event()

    def poll(self) -> PlannerUpdate | None:
        self.polled.set()
        self.released.wait(timeout=2.0)
        return PlannerUpdate(root_vel=(0.4, 0.0, 0.0))


def test_input_thread_does_not_block_poll() -> None:
    src = _BlockingInput()
    kp = KinematicPlanner(src)
    kp.start()
    try:
        # The input thread is parked inside the blocking poll(), yet the control
        # loop's poll() returns the current command immediately.
        assert src.polled.wait(timeout=1.0)
        t0 = time.perf_counter()
        cmd = kp.poll()
        assert (time.perf_counter() - t0) < 0.5
        assert cmd.root_vel == (0.0, 0.0, 0.0)  # not yet updated
        # Release the input; the command updates shortly after.
        src.released.set()
        deadline = time.time() + 2.0
        while kp.poll().root_vel == (0.0, 0.0, 0.0) and time.time() < deadline:
            time.sleep(0.01)
        assert kp.poll().root_vel == (0.4, 0.0, 0.0)
    finally:
        kp.stop()


def test_static_planner_without_input_emits_initial() -> None:
    kp = KinematicPlanner(initial=PlannerCommand(root_vel=(0.6, 0.0, 0.0)))
    kp.start()  # no-op without an input source
    try:
        assert kp.poll().root_vel == (0.6, 0.0, 0.0)
    finally:
        kp.stop()
