# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The read window a Feetech caller sets is graded, and reaches the port.

``timeout`` is how long :meth:`FeetechBus._sync_read_once` waits for the arm's
reply stream.
An unusable one does not surface as an error: pyserial accepts ``0``, ``nan``,
``inf`` and ``None`` as a timeout, and each of them makes the read see an empty
buffer that the retry loop cannot tell from a servo that never answered - so
``sync_read`` omits the motor and logs "no verified reply", and a healthy arm
reports as dead servos. That is the same failure ``baud_rate`` is graded to
prevent, in the same constructor, so the window is held to the domain every
other driver holds its timeout to.

The second half is the driver: a window it recorded in ``_extras`` instead of
forwarding is the case its own ``motor_ids`` comment names - a keyword that
changes nothing, leaving the caller believing the bus is configured.

No serial port is opened. The one cell that grades ``connect`` replaces the
bus's ``require_optional`` with a recorder, which is what makes the number the
port was opened with readable at all.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

import strands_robots.drivers.feetech.bus as bus_module
from strands_robots.drivers.feetech.bus import DEFAULT_TIMEOUT_S, FeetechBus
from strands_robots.drivers.feetech.driver import FeetechDriver
from tests.drivers.conftest import MIDPOINT_COUNTS, FakeServoPort

#: Windows no read can wait for. ``0`` returns immediately, a negative and a
#: string are what pyserial itself refuses - but from inside ``connect``, naming
#: neither the bus nor the parameter - and ``nan``/``inf``/``None``/``True``
#: pyserial takes verbatim, so nothing downstream reports them either.
UNUSABLE = [
    pytest.param(0, id="zero"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(-1.0, id="negative"),
    pytest.param("1.0", id="string"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool-reads-as-one-second"),
]

#: Windows a read can wait for: fractional, whole, and short-but-real.
USABLE = [1.0, 0.5, 2, 0.001]


class TestTheBusGradesTheReadWindow:
    """The window is refused at construction, where ``baud_rate`` is refused."""

    @pytest.mark.parametrize("timeout", UNUSABLE)
    def test_an_unusable_read_window_is_refused(self, timeout: Any) -> None:
        """The refusal names the bus, the parameter and the value handed in.

        All three, because the caller has to be able to act on it: a bare
        "invalid timeout" from inside pyserial names none of them.
        """
        with pytest.raises(ValueError) as excinfo:
            FeetechBus(port="/dev/fake", timeout=timeout)
        message = str(excinfo.value)
        assert "FeetechBus" in message
        assert "timeout" in message
        assert repr(timeout) in message or str(timeout) in message

    @pytest.mark.parametrize("timeout", USABLE)
    def test_a_usable_read_window_is_kept_as_given(self, timeout: float) -> None:
        """A positive finite window is the caller's to choose, unrounded."""
        assert FeetechBus(port="/dev/fake", timeout=timeout).timeout == timeout

    def test_the_two_layers_default_to_one_shared_window(self) -> None:
        """One number, so the driver cannot drift from the bus it builds."""
        assert FeetechBus(port="/dev/fake").timeout == DEFAULT_TIMEOUT_S
        assert FeetechDriver(tool_name="so101", port="/dev/fake").bus.timeout == DEFAULT_TIMEOUT_S


class TestTheDriverForwardsTheReadWindow:
    """A window the caller passes is honoured, not recorded."""

    def test_the_window_the_caller_asked_for_reaches_the_bus(self) -> None:
        """Forwarded to the bus, and gone from the extras bucket.

        Both halves matter: the bucket exists for keywords a downstream driver
        package may consume, and a knob this bus already has is not one of
        them.
        """
        driver = FeetechDriver(tool_name="so101", port="/dev/fake", timeout=0.25)
        assert driver.bus.timeout == 0.25
        assert "timeout" not in driver._extras

    @pytest.mark.parametrize("timeout", UNUSABLE)
    def test_an_unusable_window_is_refused_naming_the_driver(self, timeout: Any) -> None:
        """The driver names itself, as it does for ``baud_rate``.

        A refusal that named the bus would send the caller to a layer they did
        not construct.
        """
        with pytest.raises(ValueError, match=r"FeetechDriver\('so101'\): timeout"):
            FeetechDriver(tool_name="so101", port="/dev/fake", timeout=timeout)


class TestTheWindowReachesThePort:
    """The graded number is the one pyserial is opened with."""

    def test_connect_opens_the_port_with_the_window_the_caller_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grading a value the port never sees would pin nothing.

        The recorder stands in for the ``serial`` module the bus imports on
        connect, so the keyword arguments the port was opened with become
        readable without a serial stack.
        """
        opened: list[dict[str, Any]] = []

        def _serial(port: str, baud_rate: int, **kwargs: Any) -> FakeServoPort:
            opened.append({"port": port, "baud_rate": baud_rate, **kwargs})
            return FakeServoPort(MIDPOINT_COUNTS)

        fake = types.ModuleType("serial")
        fake.Serial = _serial  # type: ignore[attr-defined]
        monkeypatch.setattr(bus_module, "require_optional", lambda *_a, **_k: fake)

        driver = FeetechDriver(tool_name="so101", port="/dev/fake", timeout=0.25)
        assert driver.connect_eagerly() is None
        assert opened == [{"port": "/dev/fake", "baud_rate": 1_000_000, "timeout": 0.25}]
