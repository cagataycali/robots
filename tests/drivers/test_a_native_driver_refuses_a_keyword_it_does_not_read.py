# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A keyword a native driver does not declare is refused, not swallowed.

``Robot(name, mode="real", driver="strands", **kwargs)`` hands the keywords to
the driver class. Every driver used to end in ``**kwargs`` and park the
remainder, so a typo configured nothing and said nothing:
``Robot("so101", mode="real", driver="strands", prot="/dev/ttyACM0",
trasport="twin")`` built a ``FeetechDriver`` reporting ``port=None`` and
``transport="serial"`` - a serial arm auto-detecting a port while the caller had
named one, and a twin transport that never happened. The same two typos on
``driver="lerobot"`` are refused by name against the robot's dataclass.

The rule is the driver's own signature
(:func:`~strands_robots.drivers.base.constructor_keywords`): the keywords a
driver honours are the parameters it declares, so a driver that grows one is
graded on it with no roster to update. A ``**kwargs`` sink would re-open the
hole, which is why no shipped driver has one - pinned below over every
registered driver rather than over the three that used to read their keywords
off it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.drivers import (
    constructor_keywords,
    get_native_driver_class,
    list_native_drivers,
)
from strands_robots.drivers.feetech.driver import FeetechDriver

#: Every robot with a native driver, so the refusal is graded per robot rather
#: than per driver class - which is the granularity a caller meets it at.
_NATIVE_ROBOTS = sorted(list_native_drivers())

#: One name per driver class, for the cells whose subject is the class.
_DRIVER_CLASSES = sorted(
    {
        driver_cls.__name__: name
        for name in _NATIVE_ROBOTS
        if (driver_cls := get_native_driver_class(name)) is not None
    }.items()
)


class TestTheFactoryRefusesAKeywordTheDriverDoesNotDeclare:
    """The refusal names the keyword and the roster it is missing from."""

    @pytest.mark.parametrize("name", _NATIVE_ROBOTS)
    def test_a_typo_is_refused_naming_it_and_the_roster(self, name: str) -> None:
        driver_cls = get_native_driver_class(name)
        assert driver_cls is not None
        with pytest.raises(ValueError) as excinfo:
            Robot(name, mode="real", driver="strands", prot="/dev/ttyACM0")
        message = str(excinfo.value)
        assert "'prot'" in message, message
        assert driver_cls.__name__ in message, message
        assert "port" in message, "the roster the caller meant is the remedy"

    def test_every_declared_keyword_of_an_arm_is_accepted(self) -> None:
        """The refusal is a roster check, not a refusal of every keyword."""
        driver = Robot(
            "so101",
            mode="real",
            driver="strands",
            port="/dev/ttyACM0",
            baud_rate=500_000,
            timeout=0.5,
            motor_ids=(1, 2, 3, 4, 5, 6),
        )
        assert isinstance(driver, FeetechDriver)
        assert driver.tool_name == "so101"
        assert driver.bus.timeout == 0.5

    def test_both_drivers_refuse_the_same_typo(self) -> None:
        """The native path is the mirror of the lerobot path, not its exception."""
        pytest.importorskip("lerobot.robots.config")
        for driver in ("strands", "lerobot"):
            with pytest.raises(ValueError, match="prot"):
                Robot("so101", mode="real", driver=driver, prot="/dev/ttyACM0")


class TestTheRosterIsTheSignature:
    """Derived, so no driver can grow a keyword the factory does not know."""

    @pytest.mark.parametrize(("class_name", "name"), _DRIVER_CLASSES)
    def test_no_driver_constructor_keeps_a_kwargs_sink(self, class_name: str, name: str) -> None:
        """A sink is how the swallowing happened; the roster cannot see into one."""
        driver_cls = get_native_driver_class(name)
        assert driver_cls is not None
        sinks = [
            parameter.name
            for parameter in inspect.signature(driver_cls).parameters.values()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ]
        assert sinks == [], f"{class_name} would swallow anything outside {constructor_keywords(driver_cls)}"

    @pytest.mark.parametrize(("class_name", "name"), _DRIVER_CLASSES)
    def test_every_driver_declares_the_three_the_factory_passes(self, class_name: str, name: str) -> None:
        driver_cls = get_native_driver_class(name)
        assert driver_cls is not None
        assert {"tool_name", "cameras", "data_config"} <= set(constructor_keywords(driver_cls))

    def test_a_sink_contributes_no_keyword_and_a_parameter_does(self) -> None:
        """The two verdicts of the roster rule, on a class written for them."""

        class _Driver:
            def __init__(self, tool_name: str, *, declared: int = 0, **rest: Any) -> None:
                self._rest = rest

        assert constructor_keywords(_Driver) == ("declared", "tool_name")
