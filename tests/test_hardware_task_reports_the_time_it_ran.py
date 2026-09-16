# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A hardware task reports how long it ran, however it ended.

``RobotTaskState.duration`` is the figure every reply and every later
``status`` call reports for a finished task, and starting a task resets it to
``0.0``. Only two writers ever settled it: the rollout loop when it ran to its
own budget, and ``get_task_status`` while the task was RUNNING. Every other way
a task can end left the reset value in place, so the task reported ``0.0s``
beside a non-zero step count -- and kept reporting it for good, because nothing
writes ``duration`` once the task is terminal.

Driving the real ``_execute_task_async`` through an in-memory driver, with each
stage held for 0.4 s::

    how the task ended                      steps   reported duration
                                                     before      after
    stopped from outside (RUNNING)             39      0.0s       0.4s
    stopped during bring-up (CONNECTING)        0      0.0s       0.4s
    cleanup() latched a shutdown mid-connect    0      0.0s       0.8s
    the connect failed                          0      0.0s       0.4s
    the policy would not initialize             0      0.0s       0.4s
    the rollout raised                         39      0.0s       0.4s
    ran to its budget (control)                39      0.4s       0.4s

``error: 39 steps in 0.0s`` is the shape of the harm: an operator reading the
record of a task that failed mid-rollout, or a stop pressed on an arm that had
been moving for a minute, cannot tell it from one that ended instantly -- and
on the ``cleanup()`` and connect-failure rows the whole task WAS the bring-up,
which is the part that takes seconds on a real arm.

No serial/USB hardware is touched: the driver is an in-memory fake, the policy
is a structural stub, and the bring-up stages are coroutines that sleep.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy
from tests.test_hardware_control_loop_rate_guard import _FakeArm

#: Seconds every case holds its stage for, so a settled duration is
#: unmistakably non-zero on any host while the suite stays quick.
BUDGET = 0.4


class _Arm(_FakeArm):
    """``_FakeArm`` plus the two members ``cleanup()`` reads."""

    is_connected = True

    def disconnect(self) -> None:
        return None


class _Policy:
    """Structural policy stub, optionally failing partway through a rollout."""

    supports_rtc = False
    execution_horizon = 1

    def __init__(self, raise_after: float | None = None) -> None:
        self._raise_after = raise_after
        self._t0 = time.monotonic()

    def set_control_frequency(self, hz: float) -> None:
        return None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        return None

    def reset(self) -> None:
        return None

    async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
        if self._raise_after is not None and time.monotonic() - self._t0 >= self._raise_after:
            raise RuntimeError("servo bus went away")
        return [{"j0.pos": 0.1}]


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build real ``Robot``s whose bring-up and servo bus are in-memory."""
    made: list[Robot] = []
    monkeypatch.setattr(Robot, "_initialize_robot", lambda self, robot, cameras, **kw: _Arm())
    monkeypatch.setattr(Robot, "_migrate_legacy_calibration", lambda self: None)

    def make(*, connect_delay: float = 0.0, connect_ok: bool = True, policy_ready: bool = True) -> Robot:
        hw = Robot(tool_name="test_arm", robot="fake_arm", control_frequency=100.0)

        async def _connect() -> tuple[bool, str]:
            await asyncio.sleep(connect_delay)
            return (True, "") if connect_ok else (False, "no such port /dev/ttyACM9")

        async def _init(policy: Any) -> bool:
            await asyncio.sleep(0.0 if policy_ready else BUDGET)
            return policy_ready

        def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
            return None

        hw._connect_robot = _connect  # type: ignore[method-assign]
        hw._initialize_policy = _init  # type: ignore[method-assign]
        hw._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
        made.append(hw)
        return hw

    yield make
    for hw in made:
        hw.cleanup()


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result["content"] if "text" in c)


def _await_status(hw: Robot, status: TaskStatus, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hw._task_state.status is status:
            return
        time.sleep(0.005)
    raise AssertionError(f"task never reached {status}, stuck at {hw._task_state.status}")


def _run_in_background(hw: Robot, policy: _Policy) -> None:
    """Drive a rollout on the executor, as ``start_task`` does."""
    hw._executor.submit(hw.run_policy, cast(Policy, policy), "wave", 20.0)


def stopped_from_outside(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    hw = build()
    _run_in_background(hw, _Policy())
    _await_status(hw, TaskStatus.RUNNING)
    time.sleep(BUDGET)
    return hw, hw.stop_task(), TaskStatus.STOPPED


def stopped_during_bring_up(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    hw = build(connect_delay=BUDGET * 3)
    _run_in_background(hw, _Policy())
    _await_status(hw, TaskStatus.CONNECTING)
    time.sleep(BUDGET)
    return hw, hw.stop_task(), TaskStatus.STOPPED


def shutdown_latched_during_bring_up(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    # ``cleanup()`` calls ``stop_task()`` only for a RUNNING task, so a task
    # still connecting is ended by the rollout's own stage check instead.
    hw = build(connect_delay=BUDGET * 2)
    _run_in_background(hw, _Policy())
    _await_status(hw, TaskStatus.CONNECTING)
    time.sleep(BUDGET)
    hw.cleanup()
    return hw, hw.get_task_status(), TaskStatus.STOPPED


def connect_failed(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    hw = build(connect_delay=BUDGET, connect_ok=False)
    return hw, hw.run_policy(cast(Policy, _Policy()), "wave", 20.0), TaskStatus.ERROR


def policy_would_not_initialize(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    hw = build(policy_ready=False)
    return hw, hw.run_policy(cast(Policy, _Policy()), "wave", 20.0), TaskStatus.ERROR


def rollout_raised(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    hw = build()
    return hw, hw.run_policy(cast(Policy, _Policy(raise_after=BUDGET)), "wave", 20.0), TaskStatus.ERROR


def ran_to_its_budget(build: Any) -> tuple[Robot, dict[str, Any], TaskStatus]:
    """Control: the one ending that always reported its duration."""
    hw = build()
    return hw, hw.run_policy(cast(Policy, _Policy()), "wave", BUDGET), TaskStatus.COMPLETED


TERMINAL_TRANSITIONS = [
    stopped_from_outside,
    stopped_during_bring_up,
    shutdown_latched_during_bring_up,
    connect_failed,
    policy_would_not_initialize,
    rollout_raised,
    ran_to_its_budget,
]


@pytest.mark.parametrize("end_the_task", TERMINAL_TRANSITIONS, ids=lambda f: f.__name__)
def test_a_task_reports_the_time_it_ran_however_it_ended(build: Any, end_the_task: Any) -> None:
    """The reply and every later ``status`` carry the real elapsed time."""
    hw, reply, expected = end_the_task(build)

    assert hw._task_state.status is expected
    assert hw._task_state.duration >= BUDGET * 0.8, f"reported {hw._task_state.duration}s for a {BUDGET}s stage"
    assert "Duration: 0.0s" not in _text(reply)

    afterwards = _text(hw.get_task_status())
    assert "Total Duration: 0.0s" not in afterwards
    assert f"Total Duration: {hw._task_state.duration:.1f}s" in afterwards


def test_a_stop_reports_the_time_at_the_stop_not_at_the_last_status_call(build: Any) -> None:
    """The figure is taken when the stop happens, not inherited from a reader.

    ``get_task_status`` refreshes ``duration`` while the task is RUNNING, so a
    stop that wrote nothing itself still reported a plausible-looking number --
    the one the last status call happened to leave behind. An operator who
    checked on a task and stopped it a minute later was told it ran for the
    time it had run at the check.
    """
    hw = build()
    _run_in_background(hw, _Policy())
    _await_status(hw, TaskStatus.RUNNING)
    time.sleep(BUDGET)

    hw.get_task_status()
    at_the_check = hw._task_state.duration
    time.sleep(BUDGET)
    hw.stop_task()

    assert at_the_check >= BUDGET * 0.8
    assert hw._task_state.duration >= at_the_check + BUDGET * 0.8


def test_a_task_that_never_began_is_not_given_the_time_since_boot(build: Any) -> None:
    """Settling a state with no task behind it leaves ``duration`` alone.

    ``start_mono`` is ``0.0`` until a task begins, and subtracting that reading
    from the clock yields the machine's uptime -- days, reported as the
    duration of a task that never ran. Every current caller settles from inside
    a task, so this is pinned on the settler itself: it is the invariant a
    seventh caller has to be able to rely on, and the public surfaces cannot
    reach it (``stop`` answers an idle robot before settling anything).
    """
    hw = build()

    assert hw._task_state.start_mono == 0.0
    hw._settle_task_duration()
    assert hw._task_state.duration == 0.0
    assert _text(hw.stop_task()).startswith("No task running to stop")
