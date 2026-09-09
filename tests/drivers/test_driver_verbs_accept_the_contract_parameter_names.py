# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every shipped driver accepts the parameter names its Protocol declares.

:data:`~strands_robots.drivers.base.DRIVER_SURFACE` is derived from
:class:`~strands_robots.drivers.base.HardwareDriver` so the member *names* can
never disagree, and :func:`~strands_robots.drivers.base.missing_driver_members`
grades a candidate against it. That grader is ``hasattr``, which is the whole
gap this module closes: a driver may rename a documented parameter and satisfy
it completely, because a renamed parameter is still a present member.

The renamed parameter is not cosmetic. A driver is invoked as an agent tool and
every verb promises a status envelope - "never raise past dispatch", because an
exception there is not something the caller can handle. A caller that spells the
contract's own parameter names as keywords, which is what a dispatcher built
against :class:`HardwareDriver` does, gets a :class:`TypeError` instead. Worse
for diagnosis, a verb with ``**kwargs`` absorbs the contract's spelling
silently and then reports a *different* name missing - one the contract never
documents - so the message sends the caller looking for a parameter that is not
in the API.

The population is derived from
:data:`~strands_robots.drivers._SHIPPED_DRIVERS`, so the thirteenth driver is
held to this the hour it lands rather than inheriting an exemption by being
absent from a list.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

import strands_robots.drivers as drivers_mod
from strands_robots.drivers import (
    DRIVER_SURFACE,
    drifted_driver_parameters,
    missing_driver_members,
)
from strands_robots.drivers.dynamixel.driver import DynamixelDriver
from strands_robots.drivers.feetech import FeetechDriver


def _shipped_driver_classes() -> list[type]:
    """Every driver class the shipped table registers, deduplicated by name."""
    classes: dict[str, type] = {}
    for module_path, class_name, _names in drivers_mod._SHIPPED_DRIVERS:
        module = importlib.import_module(module_path)
        classes.setdefault(class_name, getattr(module, class_name))
    return [classes[name] for name in sorted(classes)]


SHIPPED = _shipped_driver_classes()


class TestTheWholeFleetHonoursTheContractSpelling:
    """The invariant, over every driver the package ships."""

    def test_the_population_is_not_empty(self) -> None:
        """Non-vacuity: an empty population would make every cell below pass."""
        assert len(SHIPPED) >= 10, SHIPPED

    @pytest.mark.parametrize("driver_cls", SHIPPED, ids=lambda c: c.__name__)
    def test_no_verb_renames_a_declared_parameter(self, driver_cls: type) -> None:
        assert drifted_driver_parameters(driver_cls) == ()


class TestTheTwoServoDriversThatHadDrifted:
    """The regression: these four calls raised :class:`TypeError` before the fix.

    ``run_policy`` and ``start_task`` on both servo-bus drivers spelled the
    contract's ``policy_object`` as ``policy`` and its ``instruction`` as
    ``task``. Both verbs refuse today - the control loop is a separate slice -
    and a refusal is a contract too: it has to be *reachable* under the
    documented names, so the day the loop lands no caller changes.
    """

    @pytest.mark.parametrize("driver_cls", [FeetechDriver, DynamixelDriver], ids=["feetech", "dynamixel"])
    def test_run_policy_refuses_under_the_declared_name(self, driver_cls: type) -> None:
        driver: Any = driver_cls(tool_name="arm")
        result = driver.run_policy(policy_object=None)
        assert result["status"] == "error"
        assert "run_policy" in result["content"][0]["text"]

    @pytest.mark.parametrize("driver_cls", [FeetechDriver, DynamixelDriver], ids=["feetech", "dynamixel"])
    def test_start_task_refuses_under_the_declared_name(self, driver_cls: type) -> None:
        driver: Any = driver_cls(tool_name="arm")
        result = driver.start_task(instruction="pick up the cube")
        assert result["status"] == "error"
        assert "start_task" in result["content"][0]["text"]


class TestTheGraderItself:
    """A guard that cannot fail grades nothing, so grade the guard."""

    def test_a_renamed_parameter_is_caught(self) -> None:
        """The exact drift the two servo drivers carried."""

        class Renamed:
            def run_policy(self, policy: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        drifted = dict(drifted_driver_parameters(Renamed))
        assert "run_policy" in drifted, drifted
        assert "policy" in drifted["run_policy"]

    def test_the_member_name_grader_is_blind_to_it(self) -> None:
        """Why this module exists: ``hasattr`` cannot see a renamed parameter."""

        class Renamed:
            def run_policy(self, policy: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        assert "run_policy" not in missing_driver_members(Renamed)
        assert drifted_driver_parameters(Renamed) != ()

    def test_an_instance_is_graded_like_its_class(self) -> None:
        """A caller holding a built driver gets the same answer."""
        assert drifted_driver_parameters(FeetechDriver(tool_name="arm")) == ()

    def test_extra_parameters_of_the_driver_s_own_are_allowed(self) -> None:
        """A driver may take more than the contract; only renames are refused."""

        class Extra:
            def send_action(
                self, action: dict[str, Any], robot_name: str | None = None, retries: int = 3
            ) -> dict[str, Any]:
                return {}

        assert [verb for verb, _ in drifted_driver_parameters(Extra)] == []

    def test_kwargs_may_absorb_the_parameters_a_driver_ignores(self) -> None:
        """``start_task`` declares five; absorbing four in ``**kwargs`` conforms."""

        class Absorbing:
            def start_task(self, instruction: str, **kwargs: Any) -> dict[str, Any]:
                return {}

        assert [verb for verb, _ in drifted_driver_parameters(Absorbing)] == []

    def test_a_verb_the_candidate_does_not_have_is_not_reported_here(self) -> None:
        """Absence is :func:`missing_driver_members`' verdict, not this one."""

        class Nothing:
            pass

        assert drifted_driver_parameters(Nothing) == ()
        assert set(missing_driver_members(Nothing)) == set(DRIVER_SURFACE)
