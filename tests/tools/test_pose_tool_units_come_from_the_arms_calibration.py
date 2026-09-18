# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests: this tool quotes a servo the way the Feetech driver does.

A joint's degree scale is a property of the arm, measured by ``lerobot-calibrate``
and recorded per arm. This tool used to carry a fixed per-joint table instead
(``shoulder_lift`` declared ``+-90`` across a full turn of the encoder), so the
same servo was quoted two ways depending on which of the package's two Feetech
stacks reached it: the joint moved twice as far as it was asked to and reported
half the angle it moved, and ``0 percent closed`` commanded the gripper two
thousand counts past its closed stop.

The grader here is
:class:`~strands_robots.drivers.feetech.bus.FeetechBus`, which
``tests/drivers/test_feetech_calibrated_units.py`` already grades cell for cell against
the installed LeRobot's own ``_normalize`` / ``_unnormalize``. Pinning the tool
against the bus therefore pins it against LeRobot transitively, and these cells
need no servo SDK and no arm.

The records below are what ``lerobot-calibrate`` measured on a physical SO-101
follower, so the spans are real ones: none is centred on the encoder and none
covers it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS, FeetechBus, load_calibration
from strands_robots.tools.pose_tool import MotorController, pose_tool

_PORT = "/dev/fake-arm"

#: ``Goal_Position`` on the Feetech STS/SMS control table, as ``move_motor`` writes it.
_GOAL_POSITION_ADDRESS = 0x2A

#: One physical SO-101 follower's measured travel, as ``lerobot-calibrate`` wrote it.
_MEASURED: dict[str, dict[str, int]] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -273, "range_min": 1347, "range_max": 2868},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 990, "range_min": 946, "range_max": 3302},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 73, "range_min": 677, "range_max": 2910},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -487, "range_min": 1014, "range_max": 3320},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -740, "range_min": 94, "range_max": 3982},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -480, "range_min": 2012, "range_max": 3549},
}


@pytest.fixture
def calibration_file(tmp_path: Path) -> Path:
    """The JSON ``lerobot-calibrate`` would have written for that arm."""
    path = tmp_path / "left_arm.json"
    path.write_text(json.dumps(_MEASURED))
    return path


def _driver_bus(calibration: Path | None) -> FeetechBus:
    """The authority every expectation here is derived from."""
    records = load_calibration(calibration) if calibration is not None else None
    return FeetechBus(port=_PORT, motors=SO_ARM_MOTORS, calibration=records)


def _connected(calibration: Path | None = None) -> MotorController:
    """A controller on the fake port, carrying that arm's records.

    An uncalibrated arm is built with no ``calibration`` argument at all rather
    than with ``None``, so the cells that grade the default scale grade the
    default call.
    """
    records = load_calibration(calibration) if calibration is not None else None
    controller = MotorController(_PORT) if records is None else MotorController(_PORT, calibration=records)
    connected, error = controller.connect()
    assert connected, error
    return controller


def _goal_position_written(writes: list[bytes]) -> int:
    """The count in the last ``Goal_Position`` write, decoded off the wire."""
    packet = writes[-1]
    assert packet[5] == _GOAL_POSITION_ADDRESS, f"not a Goal_Position write: {packet.hex(' ')}"
    return packet[6] | (packet[7] << 8)


def _reply(motor_id: int, counts: int) -> bytes:
    """A well-formed ``Present_Position`` status frame carrying ``counts``."""
    body = [motor_id, 4, 0, counts & 0xFF, (counts >> 8) & 0xFF]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


#: Every joint, read and driven at both ends of its measured travel and at the
#: middle - the three counts where a scale that is off by a factor shows.
_CELLS = [(name, fraction) for name in SO_ARM_MOTORS for fraction in (0.0, 0.5, 1.0)]


class TestTheArmsOwnTravelIsTheScale:
    """With the records on disk, the tool and the driver quote one number."""

    @pytest.mark.parametrize(("joint", "fraction"), _CELLS)
    def test_a_reading_is_the_value_the_driver_reports(
        self, joint: str, fraction: float, calibration_file: Path, fake_serial
    ) -> None:
        bus = _driver_bus(calibration_file)
        record = _MEASURED[joint]
        counts = round(record["range_min"] + fraction * (record["range_max"] - record["range_min"]))
        controller = _connected(calibration_file)
        fake_serial[0].queue_read(_reply(record["id"], counts))

        assert controller.read_motor_position(joint) == pytest.approx(bus.to_value(joint, counts))

    @pytest.mark.parametrize(("joint", "fraction"), _CELLS)
    def test_a_target_reaches_the_wire_as_the_driver_encodes_it(
        self, joint: str, fraction: float, calibration_file: Path, fake_serial
    ) -> None:
        bus = _driver_bus(calibration_file)
        low, high = bus.value_bounds(joint)
        target = low + fraction * (high - low)
        controller = _connected(calibration_file)

        assert controller.move_motor(joint, target) is True
        assert _goal_position_written(fake_serial[0].writes) == bus.to_counts(joint, target)

    def test_the_gripper_closes_to_its_measured_stop_not_to_encoder_zero(
        self, calibration_file: Path, fake_serial
    ) -> None:
        """``0 percent`` is where this gripper's fingers meet, and that is 2012.

        Sending encoder zero for it drove the servo two thousand counts past
        that stop - a command the arm answers by stalling against itself.
        """
        controller = _connected(calibration_file)

        assert controller.move_motor("gripper", 0.0) is True
        assert _goal_position_written(fake_serial[0].writes) == _MEASURED["gripper"]["range_min"]


class TestTheRefusalBoundsAreThatSameTravel:
    """A target is refused against the travel it would be converted through."""

    def test_a_target_past_the_measured_travel_is_refused_before_the_port_opens(
        self, calibration_file: Path, fake_serial
    ) -> None:
        low, high = _driver_bus(calibration_file).value_bounds("shoulder_pan")
        result = pose_tool(
            action="move_motor",
            port=_PORT,
            calibration=str(calibration_file),
            motor_name="shoulder_pan",
            position=high + 10.0,
        )

        assert result["status"] == "error"
        assert f"[{low:g}, {high:g}]" in _texts(result)
        assert fake_serial == [], "the port was opened for a target that could not be honored"

    def test_a_target_the_fixed_table_refused_is_honoured_on_a_calibrated_arm(
        self, calibration_file: Path, fake_serial, monkeypatch
    ) -> None:
        """This arm's ``shoulder_lift`` reaches past 100 degrees, and it may.

        The fixed table declared that joint ``+-90``, so the value LeRobot
        itself quotes for its own end stop was refused as out of range.
        """
        monkeypatch.setenv("STRANDS_POSE_COMMAND_ALLOW", "*")
        bus = _driver_bus(calibration_file)
        target = 100.0
        assert target > 90.0, "the point of this cell is a target the old table refused"

        result = pose_tool(
            action="move_motor",
            port=_PORT,
            calibration=str(calibration_file),
            motor_name="shoulder_lift",
            position=target,
        )

        assert result["status"] == "success", _texts(result)
        assert _goal_position_written(fake_serial[0].writes) == bus.to_counts("shoulder_lift", target)

    def test_a_calibration_that_is_not_one_is_reported_not_raised(self, tmp_path: Path, fake_serial) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{]")

        result = pose_tool(action="move_motor", port=_PORT, calibration=str(broken), motor_name="gripper", position=0.0)

        assert result["status"] == "error"
        assert "not JSON" in _texts(result)
        assert fake_serial == []


class TestAnUncalibratedArmIsTheServosWholeRotation:
    """With no records, every joint spans the encoder - and says so.

    The expectations here are written out rather than read from the bus, so this
    class grades the number itself: a full turn for a joint and the whole
    percent domain for the gripper. The fixed table this replaced declared 180,
    300 and 360 degrees for different joints across that same one turn, so three
    of these six joints reported a fraction of the angle they had moved.
    """

    #: The ends of the encoder, in each joint's own unit.
    _FULL_ROTATION: dict[str, tuple[float, float]] = {
        "shoulder_pan": (-180.0, 180.0),
        "shoulder_lift": (-180.0, 180.0),
        "elbow_flex": (-180.0, 180.0),
        "wrist_flex": (-180.0, 180.0),
        "wrist_roll": (-180.0, 180.0),
        "gripper": (0.0, 100.0),
    }

    @pytest.mark.parametrize("joint", sorted(SO_ARM_MOTORS))
    def test_the_top_of_the_encoder_reads_the_top_of_the_scale(self, joint: str, fake_serial) -> None:
        _low, high = self._FULL_ROTATION[joint]
        controller = _connected()
        fake_serial[0].queue_read(_reply(SO_ARM_MOTORS[joint].motor_id, SO_ARM_MOTORS[joint].resolution))

        assert controller.read_motor_position(joint) == pytest.approx(high, abs=0.1)

    @pytest.mark.parametrize("joint", sorted(SO_ARM_MOTORS))
    def test_the_bottom_of_the_scale_is_driven_to_encoder_zero(self, joint: str, fake_serial) -> None:
        low, _high = self._FULL_ROTATION[joint]
        controller = _connected()

        assert controller.move_motor(joint, low) is True
        assert _goal_position_written(fake_serial[0].writes) == 0
