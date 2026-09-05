# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One motor takes one target, whichever of its two names spells it.

``FeetechDriver.send_action`` accepts a lerobot ``"<motor>.pos"`` key and the
bare ``"<motor>"`` key for the same servo. Both vocabularies are first class in
this one driver: :func:`strands_robots.bus_access.read_joints` takes the
``bus.sync_read`` fast path for it and returns ``"<motor>.pos"`` keys, while the
tool schema (``"joint name -> degrees"``) and the success envelope's
``commanded`` block both speak the bare name. So a caller that reads the arm,
edits one joint by the name the envelope reported, and sends the result back
holds a mapping with both spellings of that motor.

Reducing the two keys to one motor is what the driver must do; doing it
silently is what it must not. A dict comprehension keyed on the stripped name
lets insertion order decide which of two targets survives, writes a frame
carrying one motor instead of two commands, and returns ``success`` naming only
the survivor - so ``{"gripper": 0.0, "gripper.pos": 100.0}`` opened the gripper
fully and reported success, and the same pair in the other order closed it.

These cells pin both halves of the refusal: the envelope says which keys name
one motor, and the write did not happen. The controls are the single-spelling
paths that must keep working, including the full read-modify-send round trip
that produces a mixed dict in the first place.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.bus_access import read_joints
from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
from tests.drivers.conftest import FakeServoPort

#: Servo ids the fake port answers for, all parked mid-scale.
_PARKED = dict.fromkeys((1, 2, 3, 4, 5, 6), 2048)


def _wired() -> FeetechDriver:
    """A driver whose bus already holds a fake port, as if connected."""
    driver = FeetechDriver(tool_name="so101", port="/dev/fake")
    driver.bus._conn = FakeServoPort(dict(_PARKED))
    return driver


def _port(driver: FeetechDriver) -> FakeServoPort:
    """The fake port behind ``driver``'s bus, narrowed to its recording surface."""
    conn = driver.bus._conn
    assert isinstance(conn, FakeServoPort)
    conn.writes.clear()
    return conn


class TestADoubledMotorIsRefused:
    """Two keys naming one motor are named back, and nothing is written."""

    @pytest.mark.parametrize(
        ("action", "motor"),
        [
            ({"gripper": 0.0, "gripper.pos": 100.0}, "gripper"),
            ({"gripper.pos": 100.0, "gripper": 0.0}, "gripper"),
            ({"gripper": 50.0, "gripper.pos": 50.0}, "gripper"),
            ({"shoulder_pan": 10.0, "shoulder_pan.pos": -10.0, "gripper": 50.0}, "shoulder_pan"),
        ],
        ids=["bare-then-suffixed", "suffixed-then-bare", "same-target-twice", "doubled-among-others"],
    )
    def test_a_motor_spelled_twice_is_refused(self, action: dict[str, Any], motor: str) -> None:
        """The refusal names the motor and both keys that spell it."""
        driver = _wired()
        _port(driver)

        result = driver.send_action(action)

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert repr(motor) in text
        for key in action:
            if key.removesuffix(".pos") == motor:
                assert repr(key) in text

    @pytest.mark.parametrize(
        "action",
        [
            {"gripper": 0.0, "gripper.pos": 100.0},
            {"gripper.pos": 100.0, "gripper": 0.0},
            {"shoulder_pan": 10.0, "shoulder_pan.pos": -10.0, "gripper": 50.0},
        ],
        ids=["bare-then-suffixed", "suffixed-then-bare", "doubled-among-others"],
    )
    def test_the_refused_command_never_reaches_the_wire(self, action: dict[str, Any]) -> None:
        """A refusal after the write would still have moved the arm."""
        driver = _wired()
        port = _port(driver)

        driver.send_action(action)

        assert port.writes == []

    def test_every_doubled_motor_is_named_not_just_the_first(self) -> None:
        """A caller fixing one collision should not have to discover the next."""
        driver = _wired()
        _port(driver)

        result = driver.send_action({"gripper": 0.0, "gripper.pos": 100.0, "wrist_roll": 5.0, "wrist_roll.pos": -5.0})

        text = result["content"][0]["text"]
        assert "'gripper'" in text and "'wrist_roll'" in text


class TestOneSpellingPerMotorStillCommandsTheArm:
    """The controls: neither vocabulary loses anything to the new refusal."""

    @pytest.mark.parametrize(
        ("action", "commanded"),
        [
            ({"shoulder_pan": 90.0, "gripper": 100.0}, {"shoulder_pan": 90.0, "gripper": 100.0}),
            ({"shoulder_pan.pos": 0.0}, {"shoulder_pan": 0.0}),
            (
                {f"{name}.pos": 0.0 for name in SO_ARM_MOTORS},
                dict.fromkeys(SO_ARM_MOTORS, 0.0),
            ),
        ],
        ids=["bare", "suffixed", "every-motor-suffixed"],
    )
    def test_a_motor_spelled_once_is_commanded(self, action: dict[str, Any], commanded: dict[str, float]) -> None:
        """The envelope reports the motors, keyed by their bare names."""
        driver = _wired()
        port = _port(driver)

        result = driver.send_action(action)

        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"] == commanded
        assert len(port.writes) == 1

    def test_a_read_sent_straight_back_is_accepted(self) -> None:
        """The documented round trip: read the arm, command the same pose.

        ``read_joints`` returns this driver's positions as ``"<motor>.pos"``,
        which is exactly one spelling per motor - so the mapping a caller reads
        is a mapping it can send.
        """
        driver = _wired()
        joints = read_joints(driver)
        assert set(joints) == {f"{name}.pos" for name in SO_ARM_MOTORS}
        port = _port(driver)

        result = driver.send_action(dict(joints))

        assert result["status"] == "success"
        assert set(result["content"][0]["json"]["commanded"]) == set(SO_ARM_MOTORS)
        assert len(port.writes) == 1
