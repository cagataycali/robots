# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``serial_tool`` refuses a Feetech parameter its packet cannot carry.

The three Feetech actions pack caller-supplied values into fixed-width fields of
a servo packet: ``motor_id`` becomes the address byte, and ``position`` /
``velocity`` are written as ``value & 0xFF`` and ``(value >> 8) & 0xFF``. That
packing *reduces* an out-of-range value instead of refusing it, so a request the
servo cannot honor became a different command it could - ``65536`` arriving as
``0``, ``-1`` as ``65535`` - while the success text echoed back the value the
caller asked for rather than the one on the bus.

The premise is pinned executably here rather than described, and the guard is
pinned to run before the port is opened: a value that cannot be carried must
never energize a motor. ``tcp_port_error``'s contract is re-pinned too, because
it now binds the same shared bounded domain instead of re-implementing it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.serial_tool import (
    _FEETECH_PARAMS,
    _PARAM_RANGES,
    feetech_param_error,
    serial_tool,
)
from strands_robots.utils import bounded_count_error, tcp_port_error

_MODULE_SRC = Path(inspect.getfile(feetech_param_error)).read_text(encoding="utf-8")


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool through one funnel.

    Several cases below deliberately supply values the tool must refuse, so they
    do not match its annotations. Routing every call through a ``**kwargs: Any``
    splat states that intent once instead of scattering per-call suppressions.
    """
    return serial_tool(**kwargs)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


# A value outside a two-byte field, and the value that would reach the bus.
WRAPPING_GOALS = [(-1, 65535), (-2048, 63488), (65536, 0), (70000, 4464)]

# ``None`` is absent rather than unusable; the required-message class owns it.
UNUSABLE_MOTOR_IDS = [0, 255, 256, -1, True, 2.7, "1"]
UNUSABLE_POSITIONS = [-1, 4096, 65536, 70000, True, 2.7, float("nan"), "2048"]
UNUSABLE_VELOCITIES = [-1, 65536, True, 2.7, "100"]


class TestThePackingPremise:
    """The bound exists because the packing cannot represent the value."""

    @pytest.mark.parametrize(("requested", "on_wire"), WRAPPING_GOALS)
    def test_a_two_byte_field_reduces_an_out_of_range_goal(self, requested: int, on_wire: int) -> None:
        """The packing this tool performs turns the request into a different value."""
        packed = (requested & 0xFF) | (((requested >> 8) & 0xFF) << 8)
        assert packed == on_wire
        assert packed != requested

    def test_every_guarded_parameter_is_bounded_by_a_field_it_must_fit(self) -> None:
        """No range accepts a value the packet could not carry."""
        for param, (minimum, maximum) in _PARAM_RANGES.items():
            assert 0 <= minimum <= maximum <= 0xFFFF, param


class TestAnUnusableParameterIsRefused:
    """A value the packet cannot carry is refused instead of reduced."""

    @pytest.mark.parametrize("motor_id", UNUSABLE_MOTOR_IDS)
    def test_motor_id_outside_the_documented_range(self, motor_id: Any, fake_serial: list) -> None:
        result = _call(action="feetech_ping", port="/dev/fake", motor_id=motor_id)
        assert result["status"] == "error"
        assert "invalid motor_id" in _text(result)
        assert "expected 1-254" in _text(result)

    @pytest.mark.parametrize("position", UNUSABLE_POSITIONS)
    def test_position_outside_the_documented_range(self, position: Any, fake_serial: list) -> None:
        result = _call(action="feetech_position", port="/dev/fake", motor_id=1, position=position)
        assert result["status"] == "error"
        assert "invalid position" in _text(result)
        assert "expected 0-4095" in _text(result)

    @pytest.mark.parametrize("velocity", UNUSABLE_VELOCITIES)
    def test_velocity_outside_the_register_width(self, velocity: Any, fake_serial: list) -> None:
        result = _call(action="feetech_velocity", port="/dev/fake", motor_id=1, velocity=velocity)
        assert result["status"] == "error"
        assert "invalid velocity" in _text(result)
        assert "expected 0-65535" in _text(result)

    def test_the_message_names_the_action_and_the_value(self, fake_serial: list) -> None:
        """A caller can act on the refusal without reading the source."""
        result = _call(action="feetech_position", port="/dev/fake", motor_id=1, position=65536)
        assert _text(result) == "feetech_position: invalid position: 65536 (expected 0-4095)"

    def test_a_wrong_type_no_longer_dead_ends_in_an_operator_error(self, fake_serial: list) -> None:
        """A float used to surface the bitwise operator that failed, naming nothing."""
        text = _text(_call(action="feetech_position", port="/dev/fake", motor_id=1, position=2.7))
        assert "unsupported operand type" not in text
        assert "bytes must be in range" not in text
        assert "invalid position" in text


class TestTheRefusalPrecedesTheBus:
    """A value that cannot be carried must never energize a motor."""

    @pytest.mark.parametrize(
        ("action", "kwargs"),
        [
            ("feetech_position", {"motor_id": 1, "position": 65536}),
            ("feetech_position", {"motor_id": 255, "position": 2048}),
            ("feetech_velocity", {"motor_id": 1, "velocity": -1}),
            ("feetech_ping", {"motor_id": 0}),
        ],
    )
    def test_a_refused_parameter_opens_no_port(self, action: str, kwargs: dict, fake_serial: list) -> None:
        result = _call(action=action, port="/dev/fake", **kwargs)
        assert result["status"] == "error"
        assert fake_serial == [], "the refusal opened the serial port"


class TestAUsableParameterStillReachesTheBus:
    """The guard must not cost a command the tool could always honor."""

    @pytest.mark.parametrize("position", [0, 1, 2048, 4095])
    def test_a_documented_position_is_written_unchanged(self, position: int, fake_serial: list) -> None:
        result = _call(action="feetech_position", port="/dev/fake", motor_id=1, position=position)
        assert result["status"] == "success"
        packet = fake_serial[0].writes[0]
        assert packet[6] | (packet[7] << 8) == position

    @pytest.mark.parametrize("motor_id", [1, 6, 253, 254])
    def test_a_documented_motor_id_addresses_that_motor(self, motor_id: int, fake_serial: list) -> None:
        result = _call(action="feetech_position", port="/dev/fake", motor_id=motor_id, position=2048)
        assert result["status"] == "success"
        assert fake_serial[0].writes[0][2] == motor_id

    @pytest.mark.parametrize("velocity", [0, 100, 3400, 65535])
    def test_a_representable_velocity_is_written_unchanged(self, velocity: int, fake_serial: list) -> None:
        result = _call(action="feetech_velocity", port="/dev/fake", motor_id=1, velocity=velocity)
        assert result["status"] == "success"
        packet = fake_serial[0].writes[0]
        assert packet[6] | (packet[7] << 8) == velocity


class TestOnlyTheParametersAnActionWritesAreChecked:
    """A caller is never refused for a value the requested action does not read."""

    @pytest.mark.parametrize("action", ["send", "read", "send_read", "list_ports"])
    def test_an_action_that_writes_no_packet_ignores_them(self, action: str) -> None:
        assert feetech_param_error(action, motor_id=999, position=99999, velocity=-1) is None

    def test_a_read_still_succeeds_with_unusable_servo_parameters(self, fake_serial: list) -> None:
        result = _call(action="read", port="/dev/fake", read_bytes=4, motor_id=999, position=-1)
        assert result["status"] == "success"

    def test_ping_ignores_a_position_it_does_not_write(self) -> None:
        assert feetech_param_error("feetech_ping", motor_id=1, position=99999, velocity=-1) is None

    def test_velocity_ignores_a_position_it_does_not_write(self) -> None:
        assert feetech_param_error("feetech_velocity", motor_id=1, position=99999, velocity=100) is None


class TestAMissingParameterIsStillReportedAsRequired:
    """Absence is not a domain error; the branch that reads it still owns it."""

    @pytest.mark.parametrize(
        ("action", "kwargs", "expected"),
        [
            ("feetech_position", {"motor_id": 1}, "motor_id and position required"),
            ("feetech_position", {"position": 2048}, "motor_id and position required"),
            ("feetech_velocity", {"motor_id": 1}, "motor_id and velocity required"),
            ("feetech_ping", {}, "motor_id required"),
        ],
    )
    def test_the_required_message_is_unchanged(
        self, action: str, kwargs: dict, expected: str, fake_serial: list
    ) -> None:
        result = _call(action=action, port="/dev/fake", **kwargs)
        assert result["status"] == "error"
        assert expected in _text(result)

    def test_an_absent_parameter_is_not_a_domain_error(self) -> None:
        assert feetech_param_error("feetech_position", motor_id=None, position=None, velocity=None) is None


class TestEveryFeetechActionIsScoped:
    """A fourth Feetech action cannot ship without declaring what it writes."""

    @staticmethod
    def _feetech_actions(source: str) -> set[str]:
        """Every ``action == "feetech_*"`` literal the module dispatches on."""
        found: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id == "action"):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    if comparator.value.startswith("feetech"):
                        found.add(comparator.value)
        return found

    def test_the_dispatched_actions_are_exactly_the_scoped_ones(self) -> None:
        assert self._feetech_actions(_MODULE_SRC) == set(_FEETECH_PARAMS)

    def test_every_scoped_parameter_has_a_range(self) -> None:
        for params in _FEETECH_PARAMS.values():
            for param in params:
                assert param in _PARAM_RANGES

    def test_the_scanner_detects_an_unscoped_action(self) -> None:
        """Without this, an empty result would read as a fully scoped module."""
        planted = _MODULE_SRC + '\nif action == "feetech_torque":\n    pass\n'
        assert self._feetech_actions(planted) - set(_FEETECH_PARAMS) == {"feetech_torque"}


class TestThePortContractIsUnchanged:
    """``tcp_port_error`` binds the shared domain; its contract must not move."""

    def test_the_message_is_byte_identical(self) -> None:
        assert tcp_port_error(0, "port", "RosbridgeRobot") == "RosbridgeRobot: invalid port: 0 (expected 1-65535)"

    @pytest.mark.parametrize("port", [1, 80, 8080, 65535])
    def test_an_addressable_port_is_still_accepted(self, port: int) -> None:
        assert tcp_port_error(port, "port", "status") is None

    @pytest.mark.parametrize("port", [0, 65536, -1, True, 2.7, "80", None])
    def test_an_unaddressable_port_is_still_refused(self, port: Any) -> None:
        assert tcp_port_error(port, "port", "status") is not None

    def test_it_agrees_with_the_domain_it_binds(self) -> None:
        for port in [0, 1, 80, 65535, 65536, -1, True, 2.7, "80", None]:
            expected = bounded_count_error(port, "port", "status", minimum=1, maximum=65535)
            assert tcp_port_error(port, "port", "status") == expected
