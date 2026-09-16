"""What a Unitree driver reports when the SDK no longer carries an IDL class.

Both Unitree drivers open their sensor topics through a plan of
``(topic, (idl_module, idl_class), decoder)`` rows and resolve each class by
importing the module and reading the attribute off it. That resolution is the
door an SDK version change arrives through: a renamed IDL class, a moved
module, or an SDK that is not installed at all reaches the driver here and
nowhere else. ``_resolve_message_class`` answers with the class or with a
reason string, and :meth:`connect_eagerly` is documented to turn that string
into a named connect failure and leave the driver "usable but not connected"
rather than raising out of bring-up.

The consumer side of that is already pinned - but with a stand-in that returns
a canned reason, so the function that produces the reason an operator actually
reads never runs. Neither refusal arm is executed on either driver, and there
is nothing holding the two copies to the same answers, which is what makes a
drift in one of them invisible.

These cells drive the real resolver on both drivers over the two ways an SDK can
fail to carry a class - the module is not importable, and the module is there
but the class is not - plus the success it has to be distinguishable from, and
then run ``connect_eagerly`` through the real resolver to check the reason
reaches the caller as a return value.

The SDK is absent on a headless runner, so a module registered on
:mod:`sys.modules` stands in for one the SDK ships; the class names come from
the drivers' own subscription plans.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_module
import strands_robots.drivers.go2 as go2_module
from strands_robots.drivers.g1 import G1Driver
from strands_robots.drivers.go2 import Go2Driver
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet

# The two copies of the resolver, and the drivers that consume them. Every cell
# below runs against both, because a reason that differs between them is a
# difference an operator reading one robot's failure cannot see in the other.
_RESOLVERS = (pytest.param(g1_module, id="g1"), pytest.param(go2_module, id="go2"))
_DRIVERS = (pytest.param(G1Driver, id="g1"), pytest.param(Go2Driver, id="go2"))

# An SDK-shaped module path, used as the module a stand-in is registered under.
# Taken from the G1's plan so the paths these cells resolve are the ones the
# driver really asks for; ``TestPremises`` holds them to that.
_SDK_MODULE = "unitree_sdk2py.idl.unitree_hg.msg.dds_"
_MISSING_MODULE = "unitree_sdk2py.idl.unitree_go.msg.dds_"
_PLAN_CLASS = "LowState_"


class _StubLowState:
    """Stands in for the IDL class the SDK would ship under ``_SDK_MODULE``."""


def _install_module(monkeypatch: pytest.MonkeyPatch, path: str, **attrs: Any) -> types.ModuleType:
    """Register a module at ``path`` carrying ``attrs``, restored on teardown.

    Args:
        monkeypatch: The requesting test's patcher, which owns the teardown.
        path: Dotted module path to register on :mod:`sys.modules`.
        attrs: Attributes to set on it - the IDL classes the module "ships".

    Returns:
        The registered module, so a cell can compare identity against it.
    """
    module = types.ModuleType(path)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, path, module)
    return module


class TestTheResolverNamesWhatItCouldNotResolve:
    """Each way a class can be unavailable gets its own reason, not a raise."""

    @pytest.mark.parametrize("driver_module", _RESOLVERS)
    def test_a_module_the_sdk_does_not_ship_is_named(self, driver_module: Any) -> None:
        """An absent module is reported with the import error, the module asked for, and the remedy.

        Every path the resolver sees is a ``unitree_sdk2py`` IDL module, so an
        absent one is the vendor SDK absent or half-installed - the reason
        carries the install recipe (:func:`sdk_missing`) and still names the
        module this call was resolving, which on a partial install is not
        always the deepest module the exception names.
        """
        reason = driver_module._resolve_message_class((_MISSING_MODULE, _PLAN_CLASS))

        assert isinstance(reason, str)
        assert reason.startswith("unitree_sdk2py is not installed: ")
        assert f"(resolving {_MISSING_MODULE})" in reason

    @pytest.mark.parametrize("driver_module", _RESOLVERS)
    def test_a_class_the_module_no_longer_carries_is_named(
        self, driver_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A module that imports but lacks the class names the class, not the import."""
        _install_module(monkeypatch, _SDK_MODULE)

        reason = driver_module._resolve_message_class((_SDK_MODULE, _PLAN_CLASS))

        assert reason == f"{_SDK_MODULE} has no {_PLAN_CLASS}"

    @pytest.mark.parametrize("driver_module", _RESOLVERS)
    def test_a_resolved_class_is_the_module_attribute_itself(
        self, driver_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The success answer is the class the bus deserialises into, not a copy."""
        module = _install_module(monkeypatch, _SDK_MODULE, **{_PLAN_CLASS: _StubLowState})

        resolved = driver_module._resolve_message_class((_SDK_MODULE, _PLAN_CLASS))

        assert resolved is getattr(module, _PLAN_CLASS)

    @pytest.mark.parametrize("driver_module", _RESOLVERS)
    def test_a_reason_is_a_string_and_a_resolved_class_is_not(
        self, driver_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``connect_eagerly`` tells the two apart by type, so the types must differ.

        Both callers discriminate with ``isinstance(message_class, str)``. A
        resolved class that answered to that test would be handed to
        ``_abort_connect`` as a failure reason, and a reason that did not would
        be handed to ``subscribe`` as a message class.
        """
        absent = driver_module._resolve_message_class((_MISSING_MODULE, _PLAN_CLASS))
        _install_module(monkeypatch, _SDK_MODULE)
        renamed = driver_module._resolve_message_class((_SDK_MODULE, _PLAN_CLASS))
        _install_module(monkeypatch, _SDK_MODULE, **{_PLAN_CLASS: _StubLowState})
        resolved = driver_module._resolve_message_class((_SDK_MODULE, _PLAN_CLASS))

        assert isinstance(absent, str)
        assert isinstance(renamed, str)
        assert not isinstance(resolved, str)


class TestBothDriversAnswerTheSameWay:
    """One drifted SDK reads the same whichever robot hit it."""

    @pytest.mark.parametrize("install", [False, True], ids=["module_absent", "class_renamed"])
    def test_the_two_resolvers_return_the_same_reason(self, install: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """The G1's and the Go2's copies produce identical text for one input."""
        path = _MISSING_MODULE
        if install:
            path = _SDK_MODULE
            _install_module(monkeypatch, path)

        assert g1_module._resolve_message_class((path, _PLAN_CLASS)) == go2_module._resolve_message_class(
            (path, _PLAN_CLASS)
        )


class TestBringUpReportsTheReasonInsteadOfRaising:
    """The reason reaches the caller as a return value, with the driver left usable."""

    @staticmethod
    def _connect(driver: Any, monkeypatch: pytest.MonkeyPatch) -> str | None:
        """Run bring-up with DDS init stubbed, so the real resolver decides.

        Args:
            driver: The driver under test.
            monkeypatch: Patcher for the one seam that needs a live bus.

        Returns:
            Whatever ``connect_eagerly`` returned.
        """
        monkeypatch.setattr(DDSSubscriberSet, "start", lambda self: None)
        return driver.connect_eagerly()

    @pytest.mark.parametrize("driver_class", _DRIVERS)
    @pytest.mark.parametrize("renamed", [False, True], ids=["module_absent", "class_renamed"])
    def test_the_resolver_reason_is_returned_and_recorded(
        self, driver_class: Any, renamed: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bring-up returns the resolver's own answer for the first plan row."""
        driver = driver_class()
        module_path, class_name = driver._subscription_plan()[0][1]
        if renamed:
            _install_module(monkeypatch, module_path)

        result = self._connect(driver, monkeypatch)

        assert isinstance(result, str)
        if renamed:
            assert result == f"{module_path} has no {class_name}"
        else:
            assert result.startswith("unitree_sdk2py is not installed: ")
            assert f"(resolving {module_path})" in result
        assert driver._connect_error == result
        assert driver._connected is False
        assert driver._subs is None

    @pytest.mark.parametrize("driver_class", _DRIVERS)
    def test_the_driver_still_answers_after_the_failed_bring_up(
        self, driver_class: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A driver left "usable but not connected" still answers its status verb."""
        driver = driver_class()

        assert isinstance(self._connect(driver, monkeypatch), str)

        status = asyncio.run(driver.get_status())
        assert status["status"] == "success"


class TestPremises:
    """What the cells above assume about the drivers they grade."""

    @pytest.mark.parametrize("driver_class", _DRIVERS)
    def test_the_plan_names_the_class_these_cells_resolve(self, driver_class: Any) -> None:
        """``_PLAN_CLASS`` is a class a driver really asks the SDK for."""
        plan = driver_class()._subscription_plan()

        assert _PLAN_CLASS in {class_name for _topic, (_module, class_name), _decoder in plan}

    @pytest.mark.parametrize("driver_class", _DRIVERS)
    def test_no_plan_module_is_importable_here(self, driver_class: Any) -> None:
        """Nothing is left registered, so the absent-module cells are not vacuous."""
        plan = driver_class()._subscription_plan()

        assert not [module for _topic, (module, _class), _decoder in plan if module in sys.modules]
