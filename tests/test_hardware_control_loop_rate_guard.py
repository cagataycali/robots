"""Behavior tests for the hardware control loop's accepted rate domain.

``strands_robots.hardware_robot.Robot`` turns ``control_frequency`` into
``action_sleep_time = 1 / control_frequency``, and that period is the only
throttle between two ``send_action`` calls on a physical servo bus. These
tests pin that a rate the loop cannot honor is refused at construction:

    - a ``0`` / negative / ``nan`` / ``inf`` / non-numeric rate raises
      ``ValueError`` naming ``control_frequency``, instead of reaching the
      division and leaving the loop unthrottled (negative / ``inf``), dying
      mid-task after the first action was already applied (``nan``), or
      surfacing a bare ``ZeroDivisionError`` / ``TypeError``;
    - the refusal happens BEFORE the lerobot driver is built, so a rejected
      rate never opens a serial port;
    - every rate that IS accepted yields the documented period, including a
      fractional and a NumPy-scalar rate;
    - the accepted domain matches the simulation's rollout-frequency domain
      (``SimEngine._validate_positive_frequency``), so the same value cannot be
      refused for a digital twin and accepted for the arm it mirrors.

No serial/USB hardware is touched: ``_initialize_robot`` is stubbed with an
in-memory fake and the calibration migration is a no-op.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from strands_robots.simulation.base import SimEngine

# Rates the loop cannot honor. ``0`` makes the period undefined; a negative or
# ``inf`` rate collapses it to a value ``asyncio.sleep`` returns from
# immediately (an unthrottled loop); ``nan`` makes ``asyncio.sleep`` raise
# after the first action has already been applied; the rest are not numbers a
# period can be computed from at all.
UNUSABLE_RATES: list[Any] = [
    0,
    0.0,
    -30.0,
    -1,
    float("inf"),
    float("-inf"),
    float("nan"),
    True,
    "30",
    None,
    [50.0],
]

# Rates that are honorable: any positive finite real, fractional or NumPy.
USABLE_RATES: list[Any] = [50.0, 30, 62.5, 0.5, 1000, np.float32(50.0), np.int64(25)]


class _FakeArm:
    """In-memory stand-in for a connected lerobot robot."""

    def __init__(self) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": {}})()

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


@pytest.fixture
def hw_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct a real ``Robot.__init__`` without touching hardware.

    Returns a callable taking the ``control_frequency`` under test plus a list
    that records every ``_initialize_robot`` call, so a test can assert the
    driver was (or was not) built.
    """
    built: list[Any] = []

    def fake_initialize_robot(self: HwRobot, robot: Any, cameras: Any, **kwargs: Any) -> _FakeArm:
        built.append(robot)
        return _FakeArm()

    monkeypatch.setattr(HwRobot, "_initialize_robot", fake_initialize_robot)
    monkeypatch.setattr(HwRobot, "_migrate_legacy_calibration", lambda self: None)

    def construct(**kwargs: Any) -> HwRobot:
        return HwRobot(tool_name="test_arm", robot="fake_arm", **kwargs)

    construct.built = built  # type: ignore[attr-defined]
    return construct


class TestUnusableRateRefused:
    @pytest.mark.parametrize("rate", UNUSABLE_RATES, ids=repr)
    def test_rate_the_loop_cannot_honor_raises(self, hw_init, rate):
        """A rate with no usable period is refused, naming the parameter."""
        with pytest.raises(ValueError, match="control_frequency"):
            hw_init(control_frequency=rate)

    @pytest.mark.parametrize("rate", UNUSABLE_RATES, ids=repr)
    def test_refusal_precedes_driver_construction(self, hw_init, rate):
        """A rejected rate never opens a serial port.

        The guard sits ahead of ``_initialize_robot``, which is what builds the
        lerobot driver and connects to the bus.
        """
        with pytest.raises(ValueError):
            hw_init(control_frequency=rate)
        assert hw_init.built == []

    def test_message_reports_the_offending_value(self, hw_init):
        with pytest.raises(ValueError) as excinfo:
            hw_init(control_frequency=-30.0)
        assert str(excinfo.value) == "Robot: control_frequency must be > 0, got -30.0."


class TestUsableRateAccepted:
    @pytest.mark.parametrize("rate", USABLE_RATES, ids=repr)
    def test_accepted_rate_yields_the_documented_period(self, hw_init, rate):
        """Every honorable rate is kept verbatim and gives period ``1 / rate``."""
        hw = hw_init(control_frequency=rate)
        try:
            assert hw.control_frequency == rate
            assert hw.action_sleep_time == pytest.approx(1.0 / float(rate))
            assert hw_init.built == ["fake_arm"]
        finally:
            hw.cleanup()

    def test_default_rate_is_accepted(self, hw_init):
        hw = hw_init()
        try:
            assert hw.action_sleep_time == pytest.approx(0.02)
        finally:
            hw.cleanup()


class TestDomainMatchesSimulation:
    @pytest.mark.parametrize("rate", UNUSABLE_RATES + USABLE_RATES, ids=repr)
    def test_hardware_and_simulation_agree_on_every_rate(self, hw_init, rate):
        """One rate domain for the arm and for the digital twin that mirrors it.

        ``SimEngine._validate_positive_frequency`` gates the simulation's
        rollout rate; the hardware loop divides by the same knob for the same
        purpose. A value accepted by one and refused by the other would mean a
        rollout that runs on the real arm cannot be rehearsed in sim, or worse.
        """
        sim_refuses = SimEngine._validate_positive_frequency(rate, "run_policy") is not None
        try:
            hw = hw_init(control_frequency=rate)
        except ValueError:
            hw_refuses = True
        else:
            hw_refuses = False
            hw.cleanup()
        assert hw_refuses == sim_refuses, f"verdicts differ for control_frequency={rate!r}"


class TestPeriodIsTheOnlyThrottle:
    """Why the guard exists: the loop has no other rate limiter.

    Pins that ``action_sleep_time`` alone bounds the command rate, so a period
    of ``<= 0`` leaves ``_execute_task_async`` free-running against the servo
    bus. Bounds are deliberately loose (an unthrottled loop overshoots by
    orders of magnitude) so the assertion holds on any host.
    """

    @staticmethod
    def _drive(period: float, duration: float = 0.2) -> int:
        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "test_arm"
        hw.action_horizon = 1
        hw.data_config = None
        hw.control_frequency = 50.0
        hw.action_sleep_time = period
        hw._task_state = RobotTaskState()
        hw._executor = ThreadPoolExecutor(max_workers=1)
        hw._shutdown_event = threading.Event()
        hw.mesh = None
        hw.peer_id = None
        hw.robot = _FakeArm()

        class _Policy:
            supports_rtc = False
            execution_horizon = 1

            def set_control_frequency(self, hz: float) -> None:
                pass

            def set_rtc_observed_delay(self, steps: int | None) -> None:
                pass

            async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
                return [{"j0.pos": 0.1}]

        async def _connected() -> tuple[bool, str]:
            return (True, "")

        async def _ready() -> bool:
            return True

        def _init_policy(policy: Any) -> Any:
            return _ready()

        def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
            return None

        hw._connect_robot = _connected  # type: ignore[method-assign]
        hw._initialize_policy = _init_policy  # type: ignore[method-assign]
        hw._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
        try:
            # ``_Policy`` is a structural stub: it provides the three members the
            # loop reads (supports_rtc / execution_horizon / get_actions) without
            # subclassing the ABC.
            asyncio.run(
                hw._execute_task_async(
                    "probe",
                    duration=duration,
                    policy_object=_Policy(),  # type: ignore[arg-type]
                )
            )
            return len(hw.robot.sent_actions)
        finally:
            hw._executor.shutdown(wait=False)

    def test_a_positive_period_throttles_the_loop(self):
        """At 50 Hz a 0.2 s task applies tens of actions, not thousands."""
        applied = self._drive(1.0 / 50.0)
        assert 0 < applied < 200

    def test_a_non_positive_period_leaves_the_loop_unthrottled(self):
        """The pathology the guard prevents: no throttle at all.

        A period of ``0`` is what ``1 / inf`` produces and what
        ``asyncio.sleep`` returns from immediately, so the loop commands the
        bus as fast as it can be driven.
        """
        applied = self._drive(0.0)
        assert applied > 1000
