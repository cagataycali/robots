"""``Robot("microduck", mode="real")`` resolves the native robotd driver.

The registry seam between Layer 1 (the microduck entry + ``hardware.driver``)
and Layer B (the driver class): a name, its aliases and an explicit
``driver="strands"`` must all build :class:`MicroduckDriver`, and adding the
hardware entry must not disturb the sim path. Construction takes no socket, so
these run with no robot attached.
"""

from __future__ import annotations

from strands_robots import Robot
from strands_robots.drivers import get_native_driver_class, list_native_drivers
from strands_robots.drivers.microduck import MicroduckDriver
from strands_robots.drivers.registry import resolve_driver
from strands_robots.registry import resolve_name

_OFF = "/tmp/microduck-not-a-real-socket.sock"


def test_microduck_registers_the_native_driver() -> None:
    assert get_native_driver_class("microduck") is MicroduckDriver
    assert list_native_drivers()["microduck"] == "MicroduckDriver"


def test_aliases_resolve_to_the_same_canonical_and_driver() -> None:
    for alias in ("micro_duck", "pollen_microduck"):
        assert resolve_name(alias) == "microduck"
        assert resolve_driver(alias) == "strands"


def test_registry_declares_strands_as_the_default_driver() -> None:
    # No explicit choice -> the robots.json hardware.driver wins, not lerobot.
    assert resolve_driver("microduck") == "strands"


def test_robot_real_mode_builds_the_microduck_driver() -> None:
    for name in ("microduck", "micro_duck"):
        driver = Robot(name, mode="real", port=_OFF)
        assert isinstance(driver, MicroduckDriver)
        assert driver.tool_name == "microduck"


def test_explicit_driver_strands_also_builds_it() -> None:
    driver = Robot("microduck", mode="real", driver="strands", port=_OFF)
    assert isinstance(driver, MicroduckDriver)


def test_sim_mode_is_untouched_by_the_hardware_entry() -> None:
    sim = Robot("microduck", mode="sim")
    # The MuJoCo engine, not a driver - adding hardware.driver must not reroute sim.
    assert not isinstance(sim, MicroduckDriver)
    assert type(sim).__name__ == "MuJoCoSimEngine"
