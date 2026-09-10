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

The bus's other traffic is graded in :mod:`test_feetech_bus`, the codec in
:mod:`test_feetech_protocol`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
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


class EchoingPort(FakeServoPort):
    """A half-duplex adapter that echoes the host's own frame before the reply.

    The frame is addressed to the servo the ack comes from, so it is not skipped
    for its ID - it is skipped because it does not verify as a status packet.
    """

    def write(self, data: bytes) -> int:
        self._pending += bytes(data)
        return super().write(data)


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
        assert len(servo_port.writes) == len(SO_ARM_MOTORS)
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
        assert len(port.writes) == len(SO_ARM_MOTORS)

    def test_stop_refuses_naming_the_joint_that_may_still_be_driven(self) -> None:
        """The driver's teardown verb reports a partial release as a refusal.

        A success envelope here tells an operator the arm is safe to approach.
        """
        driver = FeetechDriver(tool_name="so101", port="/dev/fake")
        driver.bus._conn = FakeServoPort(MUTE_ELBOW_COUNTS)
        envelope = _stop_envelope(driver)
        assert envelope["status"] == "error"
        assert "elbow_flex" in envelope["content"][0]["text"]
