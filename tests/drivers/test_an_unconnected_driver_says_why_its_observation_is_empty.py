"""An unconnected native driver's ``get_observation`` names why it is empty.

``{}`` is the no-raise answer every driver gives before its link is up, and it
also reads as a robot with no joints. ``state``, ``send_action`` and
``run_policy`` on the same driver refuse with "not connected - call
connect_eagerly() first"; the observation now logs the same cause.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.drivers.booster import BoosterDriver
from strands_robots.drivers.franka.driver import FrankaDriver
from strands_robots.drivers.kinova import KinovaDriver
from strands_robots.drivers.kuka import KukaDriver
from strands_robots.drivers.microduck import MicroduckDriver
from strands_robots.drivers.rby1 import RBY1Driver
from strands_robots.drivers.robotiq.driver import RobotiqDriver
from strands_robots.drivers.spot import SpotDriver
from strands_robots.drivers.stretch import StretchDriver
from strands_robots.drivers.ur import URDriver
from strands_robots.drivers.xarm import XArmDriver
from strands_robots.drivers.yahboom_m3pro import YahboomM3ProDriver

_DRIVERS = [
    BoosterDriver,
    lambda: FrankaDriver(tool_name="panda"),
    KinovaDriver,
    KukaDriver,
    MicroduckDriver,
    RBY1Driver,
    RobotiqDriver,
    SpotDriver,
    StretchDriver,
    URDriver,
    XArmDriver,
    YahboomM3ProDriver,
]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("make", _DRIVERS, ids=lambda m: getattr(m, "__name__", "FrankaDriver"))
def test_an_empty_observation_is_logged_with_the_call_that_connects(make, caplog) -> None:
    driver = make()
    with caplog.at_level(logging.WARNING, logger="strands_robots.drivers.base"):
        assert driver.get_observation() == {}
    assert _warnings(caplog) == [
        f"{type(driver).__name__}.get_observation: not connected - call connect_eagerly() first"
    ]


def test_a_failed_connect_is_the_reason_given(caplog) -> None:
    driver = RBY1Driver()
    reason = driver.connect_eagerly()
    assert reason is not None
    with caplog.at_level(logging.WARNING, logger="strands_robots.drivers.base"):
        assert driver.get_observation() == {}
    assert _warnings(caplog) == [f"RBY1Driver.get_observation: not connected - {reason}"]
