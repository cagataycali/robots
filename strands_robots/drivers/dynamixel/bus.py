# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Dynamixel Protocol 2.0 serial bus :class:`DynamixelDriver` writes through.

:class:`DynamixelBus` is a :class:`~strands_robots.drivers.feetech.bus.FeetechBus`
with the wire swapped: units, calibration and lifecycle are inherited unchanged,
because lerobot's ``MotorsBus`` normalises both families with the same
``_normalize`` and the same ``lerobot-calibrate`` record. What differs is the
frame (:mod:`~strands_robots.drivers.dynamixel.protocol`), the register widths
(four-byte two's-complement positions) and the torque sweep: an X-series servo
has no ``Lock`` register, so releasing is one acknowledged ``Torque_Enable``
write per motor.

The arm must already be in the operating mode lerobot configures - extended
position for the joints, current-based position for the gripper - because
``lerobot-calibrate`` writes both to EEPROM. A target outside ``0..4095``
counts is refused, not wrapped, for the reason the Feetech bus gives.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Final

from strands_robots.drivers.dynamixel.protocol import (
    CONTROL_TABLE,
    parse_status_stream,
    sync_read_packet,
    sync_write_packet,
    write_packet,
)
from strands_robots.drivers.feetech.bus import FeetechBus, MotorCalibration, MotorSpec

logger = logging.getLogger(__name__)

#: The six servos of a Koch v1.1 follower, in wire order - the IDs and norm
#: modes lerobot's ``KochFollower`` declares (``xl430-w250`` shoulder, the rest
#: ``xl330-m288``). Both models are 4096 counts a turn, the default resolution.
KOCH_MOTORS: Final[dict[str, MotorSpec]] = {
    "shoulder_pan": MotorSpec(1),
    "shoulder_lift": MotorSpec(2),
    "elbow_flex": MotorSpec(3),
    "wrist_flex": MotorSpec(4),
    "wrist_roll": MotorSpec(5),
    "gripper": MotorSpec(6, "range_0_100"),
}

#: Readable registers by the name a caller asks for, all signed on the wire.
READABLE_REGISTERS: Final[dict[str, str]] = {
    "Present_Position": "PRESENT_POSITION",
    "Present_Velocity": "PRESENT_VELOCITY",
    "Present_Current": "PRESENT_CURRENT",
}

#: Bytes of a status packet with no params: header 4, ID, LEN 2, INST, ERR, CRC 2.
_STATUS_OVERHEAD: Final[int] = 11

#: As :data:`strands_robots.drivers.feetech.bus._REPLY_SETTLE_S`: one settle per frame.
_REPLY_SETTLE_S: Final[float] = 0.01


class DynamixelBus(FeetechBus):
    """A half-duplex Dynamixel Protocol 2.0 bus carrying one arm's servos.

    Constructor as :class:`~strands_robots.drivers.feetech.bus.FeetechBus`;
    ``motors`` defaults to :data:`KOCH_MOTORS`.
    """

    NAME = "DynamixelBus"
    WIRE = "the Dynamixel bus"
    PURPOSE = "the Dynamixel Protocol 2.0 serial bus"

    def __init__(
        self,
        port: str | None,
        baud_rate: int = 1_000_000,
        motors: dict[str, MotorSpec] | None = None,
        timeout: float = 1.0,
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        super().__init__(port, baud_rate, dict(KOCH_MOTORS) if motors is None else motors, timeout, calibration)

    def sync_read(self, register: str = "Present_Position", num_retry: int = 0) -> dict[str, float]:
        """Read ``register`` from every motor in one ``SYNC_READ``, as the Feetech bus does.

        Returns:
            Motor name -> value; ``Present_Position`` in the joint's unit, the
            others as raw signed counts. Motors that did not answer are absent.

        Raises:
            ValueError: ``register`` is not readable.
            RuntimeError: The bus is not open.
        """
        if register not in READABLE_REGISTERS:
            raise ValueError(
                f"{self.NAME}: cannot read {register!r}; readable registers are {sorted(READABLE_REGISTERS)}"
            )
        conn = self._require_open(f"reading {register}")
        address, width, _ = CONTROL_TABLE[READABLE_REGISTERS[register]]
        wanted = list(dict.fromkeys(spec.motor_id for spec in self.motors.values()))
        replies: dict[int, bytes] = {}
        for _ in range(max(1, num_retry + 1)):
            if not (missing := [motor_id for motor_id in wanted if motor_id not in replies]):
                break
            conn.write(sync_read_packet(address, width, missing))
            replies.update(
                parse_status_stream(self._drain(conn, len(missing) * (_STATUS_OVERHEAD + width)), missing, width)
            )
        out: dict[str, float] = {}
        for name, spec in self.motors.items():
            if (raw := replies.get(spec.motor_id)) is None:
                logger.warning("no verified %s reply from %s (id %d)", register, name, spec.motor_id)
                continue
            value = int.from_bytes(raw, "little", signed=True)
            out[name] = self.to_value(name, value) if register == "Present_Position" else float(value)
        return out

    def write_goal_positions(self, targets: dict[str, float]) -> None:
        """Command joint positions in one ``SYNC_WRITE`` of four-byte ``Goal_Position`` words.

        Raises:
            ValueError: As :meth:`FeetechBus.write_goal_positions`.
            RuntimeError: The bus is not open.
        """
        conn, counts = self._goal_counts(targets)
        address, width, _ = CONTROL_TABLE["GOAL_POSITION"]
        conn.write(
            sync_write_packet(address, width, [(i, c.to_bytes(width, "little", signed=True)) for i, c in counts])
        )

    def set_torque(self, enabled: bool) -> list[str]:
        """Energize or release every motor with an acknowledged ``Torque_Enable`` write.

        Every motor is attempted even after one fails, and a motor whose ack did
        not verify is returned: after ``enabled=False`` a non-empty list means
        those joints may still be driven.

        Raises:
            RuntimeError: The bus is not open.
        """
        conn = self._require_open("setting torque")
        address = CONTROL_TABLE["TORQUE_ENABLE"][0]
        failed: list[str] = []
        for name, spec in self.motors.items():
            try:
                conn.write(write_packet(spec.motor_id, address, bytes([1 if enabled else 0])))
                raw = self._drain(conn, _STATUS_OVERHEAD)
            except OSError as e:
                logger.error("failed to write TORQUE_ENABLE on %s (id %d): %s", name, spec.motor_id, e)
                failed.append(name)
                continue
            if spec.motor_id not in parse_status_stream(raw, [spec.motor_id], 0):
                logger.error("no verified TORQUE_ENABLE ack from %s (id %d): %s", name, spec.motor_id, raw.hex(" "))
                failed.append(name)
        return failed

    @staticmethod
    def _drain(conn: Any, size: int) -> bytes:
        """Read the ``size`` bytes a reply carries, plus any echo still buffered in front of it."""
        time.sleep(_REPLY_SETTLE_S)
        raw = bytes(conn.read(size))
        if echoed := int(getattr(conn, "in_waiting", 0) or 0):
            raw += bytes(conn.read(echoed))
        return raw
