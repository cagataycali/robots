# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A torque write is answered, and :meth:`FeetechBus.set_torque` reads the answer.

A unicast ``WRITE`` on the Feetech bus returns a six-byte status packet. That
reply is the only evidence a servo took the command, so it is what
:meth:`~strands_robots.drivers.feetech.driver.FeetechDriver._set_torque_envelope`
means when it refuses with "these motors did not answer and may still be
driven" - and it is also six frames the next reader has to get past.

Both halves are graded here: a servo that did not answer is named rather than
assumed released, and the state read that follows a torque sweep sees its own
frames instead of the sweep's leftovers.

Which registers the sweep writes is graded here too, against LeRobot's own bus:
``Torque_Enable`` and ``Lock`` carry the same value on this series, and an arm
energized without the second one runs with its EEPROM open to any malformed
frame.

The bus's other traffic is graded in :mod:`test_feetech_bus`, the codec in
:mod:`test_feetech_protocol`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
from strands_robots.drivers.feetech.protocol import Register
from tests.drivers.conftest import MIDPOINT_COUNTS, FakeServoPort, open_bus

#: The arm with its elbow unplugged: servo 3 is on the bus map but answers
#: nothing, which is what a loose connector looks like from the host.
MUTE_ELBOW_COUNTS = {motor_id: counts for motor_id, counts in MIDPOINT_COUNTS.items() if motor_id != 3}


class PortWithoutInWaiting(FakeServoPort):
    """A port object that does not report its buffer.

    :meth:`FeetechBus._sync_read_once` documents support for one ("a port
    without that attribute simply skips the top-up"), so it is the port that
    shows what unread acks in front of a reply stream cost: without a top-up to
    recover them, the bytes they displaced are gone.
    """

    @property
    def in_waiting(self) -> int:
        return 0


class TruncatingPort(FakeServoPort):
    """Every ack arrives one byte short, so no frame verifies.

    A servo that answered something unusable is not distinguishable from one
    that stayed silent, and both mean the same thing: nothing confirmed the
    write.
    """

    def read(self, size: int) -> bytes:
        return super().read(size)[:-1]


class PortThatFailsOnOneMotor(FakeServoPort):
    """The host's own port stops working part-way through a sweep.

    ``set_torque`` writes one frame per servo, so a port that raises is a
    different failure from a servo that stays silent: nothing was put on the
    wire for that motor at all. Both must land in the same return, because the
    caller's question - "is this joint still driven?" - has the same answer.

    Args:
        counts: As :class:`FakeServoPort`.
        failing_id: The motor whose frame raises ``OSError``. pyserial raises
            ``SerialException``, a subclass, so one clause covers both.
    """

    def __init__(self, counts: dict[int, int], *, failing_id: int) -> None:
        super().__init__(counts)
        self.failing_id = failing_id

    def write(self, data: bytes) -> int:
        if data[2] == self.failing_id:
            raise OSError(5, "Input/output error")
        return super().write(data)


class EchoingPort(FakeServoPort):
    """A half-duplex adapter that echoes the host's own frame before the reply.

    The frame is addressed to the servo the ack comes from, so it is not skipped
    for its ID - it is skipped because it does not verify as a status packet.
    """

    def write(self, data: bytes) -> int:
        self._pending += bytes(data)
        return super().write(data)


class PortThatIgnoresTheLockWrite(FakeServoPort):
    """Every servo acks its torque write and none acks its ``Lock`` write.

    The two writes answer different questions, so they cannot share one verdict:
    the joint is in the energization state it was asked for - that write was
    acknowledged - and only its EEPROM write protection is unknown.
    """

    def write(self, data: bytes) -> int:
        if data[4] == 0x03 and data[5] == Register.LOCK:
            self.writes.append(bytes(data))
            return len(data)
        return super().write(data)


def _written_registers(port: FakeServoPort) -> list[tuple[int, int, int]]:
    """Decode a recorded ``WRITE`` sweep into ``(motor_id, address, value)``."""
    return [(frame[2], frame[5], frame[6]) for frame in port.writes if frame[4] == 0x03]


def _lerobot_torque_registers(*, enabled: bool) -> list[tuple[int, int, int]]:
    """The same triples LeRobot's own Feetech bus puts on the wire.

    The oracle is driven in-process: LeRobot's bus constructs without a port,
    and its vendor write is replaced by a recorder, so what is compared is the
    register sequence rather than a restatement of it in this file.
    """
    feetech = pytest.importorskip("lerobot.motors.feetech")
    motors_module = pytest.importorskip("lerobot.motors")
    bus = feetech.FeetechMotorsBus(
        port="/dev/null",
        motors={
            name: motors_module.Motor(spec.motor_id, "sts3215", motors_module.MotorNormMode.RANGE_M100_100)
            for name, spec in SO_ARM_MOTORS.items()
        },
    )
    recorded: list[tuple[int, int, int]] = []

    def record(_port: Any, motor_id: int, address: int, _length: int, value: Any) -> tuple[int, int]:
        # Both registers are one byte wide; LeRobot's vendor call hands the
        # payload over as a sequence, so it is unwrapped to the byte it carries.
        recorded.append((motor_id, address, int(value) if isinstance(value, int) else int(value[0])))
        return (0, 0)

    bus.port_handler.is_open = True
    bus.packet_handler.writeTxRx = record
    bus.enable_torque() if enabled else bus.disable_torque()
    return recorded


def _stop_envelope(driver: FeetechDriver) -> dict[str, Any]:
    """Run the driver's ``stop`` verb and return the tool envelope it yields."""

    async def drive() -> dict[str, Any]:
        result: dict[str, Any] = {}
        async for event in driver.stream({"toolUseId": "t", "name": "so101", "input": {"action": "stop"}}, {}):
            if isinstance(event, dict) and "status" in event:
                result = event
        return result

    return asyncio.run(drive())


class TestAnAnsweredWriteIsReadBack:
    """The ack is consumed, so it is neither assumed nor left on the bus."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_a_healthy_arm_answers_every_torque_write(self, servo_port: FakeServoPort, enabled: bool) -> None:
        """Six writes, six acks read: nothing failed and nothing is left buffered."""
        bus = open_bus(servo_port)
        assert bus.set_torque(enabled) == []
        assert len(servo_port.writes) == 2 * len(SO_ARM_MOTORS), "torque and Lock, per motor"
        assert servo_port.in_waiting == 0, "the acks are still in front of the next reader's frame"

    def test_the_state_read_after_a_sweep_sees_its_own_frames(self) -> None:
        """The read following a sweep answers with the whole arm.

        Six unread acks are 36 bytes ahead of a ``SYNC_READ`` reply stream, and
        a port with nothing to top up from cannot get them back: the reply is
        read short and five healthy servos report as not answering.
        """
        port = PortWithoutInWaiting(MIDPOINT_COUNTS)
        bus = open_bus(port)
        bus.set_torque(True)
        assert sorted(bus.sync_read("Present_Position")) == sorted(SO_ARM_MOTORS)

    def test_the_hosts_own_echo_in_front_of_the_ack_is_skipped(self) -> None:
        """An echoing adapter is not a mute arm: the ack behind the echo counts."""
        bus = open_bus(EchoingPort(MIDPOINT_COUNTS))
        assert bus.set_torque(False) == []


class TestASilentServoIsNamed:
    """ "Did not answer" is measured, because the caller is told the joint may still be driven."""

    @pytest.mark.parametrize(
        ("port", "expected"),
        [
            pytest.param(FakeServoPort(MUTE_ELBOW_COUNTS), ["elbow_flex"], id="one-servo-mute"),
            pytest.param(TruncatingPort(MIDPOINT_COUNTS), list(SO_ARM_MOTORS), id="no-frame-verifies"),
        ],
    )
    def test_a_servo_that_did_not_answer_is_reported_not_assumed_released(
        self, port: FakeServoPort, expected: list[str]
    ) -> None:
        """Named in the return, and every other motor was still attempted."""
        bus = open_bus(port)
        assert bus.set_torque(False) == expected
        # A torque write that was not acknowledged stops that motor there: no
        # Lock frame follows a joint whose energization is unconfirmed.
        assert len(port.writes) == 2 * len(SO_ARM_MOTORS) - len(expected)

    def test_a_port_that_fails_mid_sweep_names_that_joint_and_keeps_going(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The twin of the mute servo above, one layer lower.

        A release that gave up at the first failure would leave the joints
        after it energized while reporting only the one it stopped on - so the
        motors past the failure are the assertion, not the failure itself.
        """
        bus = open_bus(PortThatFailsOnOneMotor(MIDPOINT_COUNTS, failing_id=SO_ARM_MOTORS["elbow_flex"].motor_id))
        with caplog.at_level(logging.ERROR):
            assert bus.set_torque(False) == ["elbow_flex"]
        port = bus._conn
        assert isinstance(port, PortThatFailsOnOneMotor)
        # Five motors' pairs for six motors: the elbow's torque frame never
        # reached the wire, so neither did its Lock frame, and the gripper's -
        # the last on the bus - did.
        assert len(port.writes) == 2 * (len(SO_ARM_MOTORS) - 1)
        assert port.writes[-1][2] == SO_ARM_MOTORS["gripper"].motor_id
        assert "failed to write TORQUE_ENABLE on elbow_flex" in caplog.text

    def test_stop_refuses_naming_the_joint_that_may_still_be_driven(self) -> None:
        """The driver's teardown verb reports a partial release as a refusal.

        A success envelope here tells an operator the arm is safe to approach.
        """
        driver = FeetechDriver(tool_name="so101", port="/dev/fake")
        driver.bus._conn = FakeServoPort(MUTE_ELBOW_COUNTS)
        envelope = _stop_envelope(driver)
        assert envelope["status"] == "error"
        assert "elbow_flex" in envelope["content"][0]["text"]


class TestTorqueAndLockAreOneDecision:
    """``Lock`` rides with ``Torque_Enable``, because the servo ties them together.

    ``lerobot-calibrate`` leaves an arm with ``Lock`` clear - LeRobot's
    ``disable_torque`` clears it so the calibration can be written. An energize
    that wrote ``Torque_Enable`` alone then drove that arm for the whole rollout
    with its EEPROM - ID, baud rate, position limits - open to any malformed
    frame.
    """

    @pytest.mark.parametrize("enabled", [True, False])
    def test_each_motor_gets_torque_then_lock_carrying_the_same_value(
        self, servo_port: FakeServoPort, enabled: bool
    ) -> None:
        """Two frames per motor, in that order, both holding the same byte."""
        bus = open_bus(servo_port)
        assert bus.set_torque(enabled) == []
        value = 1 if enabled else 0
        assert _written_registers(servo_port) == [
            pair
            for spec in SO_ARM_MOTORS.values()
            for pair in (
                (spec.motor_id, Register.TORQUE_ENABLE, value),
                (spec.motor_id, Register.LOCK, value),
            )
        ]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_the_sweep_writes_what_lerobot_writes_for_the_same_call(
        self, servo_port: FakeServoPort, enabled: bool
    ) -> None:
        """The same arm, driven by either stack, is left in the same servo state.

        LeRobot is the reference an SO-arm is calibrated and driven by, so a
        register it writes and this bus does not is a divergence in what the
        servo is left holding - not a style difference.
        """
        bus = open_bus(servo_port)
        assert bus.set_torque(enabled) == []
        assert _written_registers(servo_port) == _lerobot_torque_registers(enabled=enabled)

    def test_a_servo_that_ignores_its_lock_write_is_logged_not_called_driven(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A release is still a release: the return names joints that may move.

        Naming this joint would tell an operator the arm is unsafe to approach
        over a write protection flag, which is the opposite of the refusal's
        purpose.
        """
        bus = open_bus(PortThatIgnoresTheLockWrite(MIDPOINT_COUNTS))
        with caplog.at_level(logging.ERROR):
            assert bus.set_torque(False) == []
        assert "EEPROM write protection is now unknown" in caplog.text
