# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Feetech serial bus :class:`FeetechDriver` writes through.

:mod:`~strands_robots.drivers.feetech.protocol` builds and parses the frames -
including the two-byte word order, which is a property of the STS/SMS series and
not of the family, so it is read from there rather than spelled again here. This
module puts those frames on a wire and turns the two byte values a servo speaks
into the units a caller uses. It is the half :issue:`360` scope 1 named as
deferred when the driver landed as a stub.

Units, stated once because a wrong unit is a wrong motion and not an error:
every joint is **degrees**, and ``gripper`` is **percent open** (0 closed,
100 open). That is the domain :mod:`strands_robots.tools.pose_tool` already
established for this arm family, and its ``MotorConfig`` ranges are the
source of truth :data:`SO_ARM_MOTORS` is distilled from - the same six IDs in
the same wire order that lerobot's ``SOFollower`` drives.

A target outside a joint's range is **refused, not clamped**. A clamp turns a
caller's 400-degree command into a 180-degree motion and reports success, so
the caller learns nothing and the arm goes somewhere it was not told to go.

Nothing here imports :mod:`serial` at module load: the import happens in
:meth:`FeetechBus.connect`, so the package stays importable - and its codec
gradeable - on a box with no serial stack at all.
"""

from __future__ import annotations

import logging
import math
import numbers
import time
from typing import Any, Final

from strands_robots.drivers.feetech.protocol import (
    MAX_GOAL_POSITION,
    SIGN_BIT,
    STATUS_OVERHEAD,
    Register,
    decode_sign_magnitude,
    decode_word,
    encode_word,
    parse_sync_read_replies,
    sync_read_packet,
    sync_read_reply_size,
    sync_write_packet,
    write_packet,
)
from strands_robots.utils import positive_count_error, positive_finite_number_error, require_optional

logger = logging.getLogger(__name__)


class MotorSpec:
    """One servo on the bus: its ID and the caller-facing range it spans.

    Attributes:
        motor_id: The servo's ID on the shared half-duplex bus.
        low: Lowest caller value, in the joint's own unit.
        high: Highest caller value, in the joint's own unit.
        resolution: Encoder counts spanning ``low``..``high``. Defaults to
            :data:`~strands_robots.drivers.feetech.protocol.MAX_GOAL_POSITION`,
            the STS/SMS full scale - an SCS-series servo spans a quarter of it
            and must be given its own.
    """

    __slots__ = ("high", "low", "motor_id", "resolution")

    def __init__(self, motor_id: int, low: float, high: float, resolution: int = MAX_GOAL_POSITION) -> None:
        self.motor_id = motor_id
        self.low = low
        self.high = high
        self.resolution = resolution

    def to_counts(self, value: float) -> int:
        """Encode a caller value as encoder counts.

        Args:
            value: The target, in this joint's unit.

        Returns:
            Counts in ``0..resolution``.

        Raises:
            ValueError: When ``value`` is outside ``low``..``high``. Refused
                rather than clamped - see the module docstring.
        """
        if not self.low <= value <= self.high:
            raise ValueError(f"target {value} outside range ({self.low}, {self.high})")
        span = self.high - self.low
        return round((value - self.low) / span * self.resolution)

    def to_value(self, counts: int) -> float:
        """Decode encoder counts back into this joint's unit.

        A reading is not bounded the way a target is. A servo whose homing
        offset puts the joint just past its zero reports negative counts, and
        the value they map to sits below :attr:`low` - reported as it is,
        because that is where the joint actually is. Refusing or clamping it
        would discard the arm's own state rather than a caller's mistake.
        """
        span = self.high - self.low
        return self.low + counts / self.resolution * span


#: The six servos of an SO-100 / SO-101 follower, in wire order.
#:
#: IDs, ranges and the gripper's percent domain are carried over verbatim from
#: :data:`strands_robots.tools.pose_tool._DEFAULT_MOTOR_CONFIGS`, which is the
#: map proven against the physical arm. Keeping one map for the tool and the
#: driver is what stops the two commanding different joints by the same name.
SO_ARM_MOTORS: Final[dict[str, MotorSpec]] = {
    "shoulder_pan": MotorSpec(1, -180, 180),
    "shoulder_lift": MotorSpec(2, -90, 90),
    "elbow_flex": MotorSpec(3, -150, 150),
    "wrist_flex": MotorSpec(4, -90, 90),
    "wrist_roll": MotorSpec(5, -180, 180),
    "gripper": MotorSpec(6, 0, 100),
}

#: Readable registers, keyed by the name a caller asks for.
#:
#: The value is the register and nothing else. Whether it carries a sign, and on
#: which bit, is looked up in
#: :data:`~strands_robots.drivers.feetech.protocol.SIGN_BIT` - not restated
#: here, because restating it is how ``Present_Position`` came to be read as
#: unsigned while the two registers either side of it were not.
#:
#: ``Present_Current`` (0x45) is absent because nothing in this package reads
#: it, not because its encoding is unknown: lerobot's encodings table carries no
#: entry for it, which is to say unsigned, so adding it is one line here and
#: none there. A caller asking for it gets a refusal naming the readable set.
READABLE_REGISTERS: Final[dict[str, Register]] = {
    "Present_Position": Register.PRESENT_POSITION,
    "Present_Velocity": Register.PRESENT_VELOCITY,
    "Present_Load": Register.PRESENT_LOAD,
}

#: Bytes each readable register carries.
_REGISTER_WIDTH: Final[int] = 2

#: Param bytes a servo's answer to a ``WRITE`` carries: none. The frame is the
#: whole reply, which is why the ack read asks the port for
#: :data:`~strands_robots.drivers.feetech.protocol.STATUS_OVERHEAD` bytes.
_ACK_PARAM_COUNT: Final[int] = 0

#: Seconds a read waits for a servo's reply. Named rather than spelled
#: twice: :class:`~strands_robots.drivers.feetech.driver.FeetechDriver`
#: forwards a caller's window to this bus and defaults to the same one.
DEFAULT_TIMEOUT_S: Final[float] = 1.0

#: Seconds to let the arm answer before reading. The vendor SDK polls; a fixed
#: settle keeps the read simple and is ample at 1 Mbaud, where a whole six-servo
#: ``SYNC_READ`` reply stream is 48 bytes - under half a millisecond on the wire.
#: One settle covers the whole arm because one frame asks the whole arm; paying
#: it per servo is what put a floor of ``motors * 10 ms`` under every state read.
_REPLY_SETTLE_S: Final[float] = 0.01


def _decode(raw: bytes, sign_bit: int | None) -> int:
    """Turn a two-byte reply into the number the servo meant.

    Both halves come from the codec:
    :func:`~strands_robots.drivers.feetech.protocol.decode_word` for the byte
    order and
    :func:`~strands_robots.drivers.feetech.protocol.decode_sign_magnitude` for
    the sign, so this bus holds no copy of either convention.

    Args:
        raw: The register's parameter bytes, in the order the servo sent them.
        sign_bit: The register's
            :data:`~strands_robots.drivers.feetech.protocol.SIGN_BIT` entry, or
            ``None`` when the register is unsigned.

    Returns:
        The register value, negative when ``sign_bit`` is set in it.
    """
    value = decode_word(raw)
    return value if sign_bit is None else decode_sign_magnitude(value, sign_bit)


class FeetechBus:
    """A half-duplex Feetech bus carrying one SO-arm's servos.

    Args:
        port: Serial device path. ``None`` is accepted so a driver can be
            constructed before its port is known; :meth:`connect` refuses.
        baud_rate: Bus speed, a positive integer. 1 Mbaud is the STS3215
            default. Refused here rather than at :meth:`connect` because
            pyserial coerces the speed through its own ``int()`` and refuses
            only a negative, so an unusable value opens the port at a speed no
            servo answers instead of reporting itself.
        motors: The servos on this bus, defaulting to :data:`SO_ARM_MOTORS`.
        timeout: How long a read waits for a servo's reply, in seconds;
            :data:`DEFAULT_TIMEOUT_S` unless a caller knows the bus answers
            slower. Held to the same domain as ``baud_rate`` and for the same
            reason: pyserial accepts ``0``, ``nan``, ``inf`` and ``None`` as a
            timeout, and every one of them makes :meth:`_sync_read_once` see an
            empty buffer it cannot tell from an arm that never answered - so a
            healthy arm reports as motors that did not reply, naming neither
            this bus nor the value that decided it. The two pyserial does
            refuse it refuses from inside :meth:`connect`, naming neither.

    Raises:
        ValueError: ``baud_rate`` is not a positive integer, or ``timeout`` is
            not a positive finite number.
    """

    def __init__(
        self,
        port: str | None,
        baud_rate: int = 1_000_000,
        motors: dict[str, MotorSpec] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if (reason := positive_count_error(baud_rate, "baud_rate", type(self).__name__)) is not None:
            raise ValueError(reason)
        if (reason := positive_finite_number_error(timeout, "timeout", type(self).__name__)) is not None:
            raise ValueError(reason)
        self.port = port
        self.baud_rate = baud_rate
        self.motors = dict(SO_ARM_MOTORS if motors is None else motors)
        self.timeout = timeout
        self._conn: Any | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """Whether the port is open, so a caller can tell live from stale."""
        conn = self._conn
        return bool(conn is not None and getattr(conn, "is_open", False))

    def connect(self) -> None:
        """Open the serial port.

        Raises:
            ValueError: When no port was configured.
            OSError: When the port cannot be opened. ``serial.SerialException``
                subclasses ``OSError``, so one except clause covers both.
        """
        if self.is_connected:
            return
        if not self.port:
            raise ValueError("FeetechBus: no port configured; pass port= to open the SCS bus")
        serial = require_optional(
            "serial",
            pip_install="pyserial",
            purpose="the Feetech SCS serial bus",
        )
        self._conn = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)  # type: ignore[attr-defined]

    def disconnect(self) -> None:
        """Close the port. Safe to call when already closed."""
        conn, self._conn = self._conn, None
        if conn is not None and getattr(conn, "is_open", False):
            conn.close()

    def _require_open(self, what: str) -> Any:
        """Return the open connection, or raise naming what was attempted."""
        if not self.is_connected:
            raise RuntimeError(f"FeetechBus: {what} needs an open bus; call connect() first (port={self.port!r})")
        return self._conn

    # ------------------------------------------------------------------ #
    # Reads.                                                              #
    # ------------------------------------------------------------------ #

    def sync_read(self, register: str = "Present_Position", num_retry: int = 0) -> dict[str, float]:
        """Read ``register`` from every motor, in one ``SYNC_READ`` frame.

        Named and shaped for :func:`strands_robots.bus_access.read_joints`,
        which calls ``bus.sync_read("Present_Position")`` and appends the
        ``.pos`` suffix itself - so exposing this method is what puts an
        SO-arm's joints on the mesh state topic.

        One frame for the whole arm rather than one per joint, for the reason
        :meth:`write_goal_positions` gives on the write side: the servos answer
        the same packet, so the six numbers are one pose taken at one instant
        instead of six samples smeared across as many round trips. A per-motor
        READ also pays the reply settle once per servo, which put a floor of
        ``motors * 10 ms`` under every state read - 60 ms on a six-servo arm,
        below the 30 Hz the joint consumers named in
        :func:`~strands_robots.bus_access.read_joints` publish at.

        Every servo this bus carries answers ``SYNC_READ``: the instruction is
        unavailable on the SCS series, which is protocol 1 and which this codec
        does not address at all (see
        :mod:`~strands_robots.drivers.feetech.protocol`), and lerobot keys the
        same refusal on the same per-model protocol number.

        A motor whose reply does not verify is omitted rather than guessed at,
        exactly as
        :meth:`strands_robots.tools.pose_tool.MotorController.read_all_positions`
        omits it.

        Args:
            register: A key of :data:`READABLE_REGISTERS`.
            num_retry: Extra attempts before giving up. Each retry re-asks only
                the motors still missing, so a mute servo costs the retries it
                is given and the rest of the arm costs none.

        Returns:
            Motor name -> value. ``Present_Position`` is in the joint's own
            unit (degrees, or percent for ``gripper``); the other registers are
            raw signed counts. Motors that did not answer are absent.

        Raises:
            ValueError: When ``register`` is not readable.
            RuntimeError: When the bus is not open.
        """
        if register not in READABLE_REGISTERS:
            raise ValueError(
                f"FeetechBus: cannot read {register!r}; readable registers are {sorted(READABLE_REGISTERS)}"
            )
        conn = self._require_open(f"reading {register}")
        address = READABLE_REGISTERS[register]
        sign_bit = SIGN_BIT.get(address)
        # Keyed by ID, so a bus that names one servo twice asks for it once.
        wanted = list(dict.fromkeys(spec.motor_id for spec in self.motors.values()))
        replies: dict[int, bytes] = {}
        for _ in range(max(1, num_retry + 1)):
            missing = [motor_id for motor_id in wanted if motor_id not in replies]
            if not missing:
                break
            replies.update(self._sync_read_once(conn, address, missing))

        out: dict[str, float] = {}
        for name, spec in self.motors.items():
            raw = replies.get(spec.motor_id)
            if raw is None:
                logger.warning("no verified %s reply from %s (id %d)", register, name, spec.motor_id)
                continue
            value = _decode(raw, sign_bit)
            out[name] = spec.to_value(value) if register == "Present_Position" else float(value)
        return out

    def _sync_read_once(self, conn: Any, address: int, motor_ids: list[int]) -> dict[int, bytes]:
        """Ask ``motor_ids`` for ``address`` once, returning the replies that verified.

        The port is asked for exactly the bytes the reply stream carries
        (:func:`~strands_robots.drivers.feetech.protocol.sync_read_reply_size`).
        Asking for more waits out the whole read window for bytes no servo is
        going to send, which is how a healthy arm comes to read at the timeout
        instead of at the wire. ``in_waiting`` is then drained so that echoed
        bytes in front of the first frame do not push the last servo's frame
        past the count - a port without that attribute simply skips the top-up.
        """
        conn.write(sync_read_packet(address, _REGISTER_WIDTH, motor_ids))
        time.sleep(_REPLY_SETTLE_S)
        raw = bytes(conn.read(sync_read_reply_size(len(motor_ids), _REGISTER_WIDTH)))
        echoed = int(getattr(conn, "in_waiting", 0) or 0)
        if echoed:
            raw += bytes(conn.read(echoed))
        return parse_sync_read_replies(raw, motor_ids, _REGISTER_WIDTH)

    # ------------------------------------------------------------------ #
    # Writes.                                                             #
    # ------------------------------------------------------------------ #

    def write_goal_positions(self, targets: dict[str, float]) -> None:
        """Command joint positions, all in one SYNC_WRITE frame.

        One frame for the whole arm rather than one per joint: the servos then
        latch their goals from the same packet, so a six-joint move starts
        together instead of smearing over six write latencies.

        Args:
            targets: Motor name -> target, in the joint's own unit.

        Raises:
            ValueError: When a name is not on this bus, a value is not finite,
                or a value is outside the joint's range.
            RuntimeError: When the bus is not open.
        """
        if not targets:
            raise ValueError("FeetechBus: no targets to write")
        conn = self._require_open("writing goal positions")
        motor_data: list[tuple[int, bytes]] = []
        for name, value in targets.items():
            spec = self.motors.get(name)
            if spec is None:
                raise ValueError(f"FeetechBus: unknown motor {name!r}; this bus carries {sorted(self.motors)}")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"FeetechBus: {name} target must be a finite number, got {type(value).__name__}: {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"FeetechBus: {name} target must be finite, got {value!r}")
            counts = spec.to_counts(number)
            motor_data.append((spec.motor_id, encode_word(counts)))
        conn.write(sync_write_packet(Register.GOAL_POSITION, _REGISTER_WIDTH, motor_data))

    def set_torque(self, enabled: bool) -> list[str]:
        """Energize or release every motor, returning the ones that failed.

        A unicast ``WRITE`` is answered - the servo returns the empty status
        packet :func:`~strands_robots.drivers.feetech.protocol.write_packet`
        documents - and that reply is read back here, for two reasons.

        It is the only evidence the motor took the command. Without it the sole
        failure this could report is an ``OSError`` from the host's own port, so
        a servo that is unplugged, mute, or answering garbage is reported as
        released; the refusal
        :meth:`~strands_robots.drivers.feetech.driver.FeetechDriver._set_torque_envelope`
        raises on a non-empty return names those motors as possibly still
        driven, and a claim about a joint that may still be moving is worth
        measuring rather than assuming.

        And the acks are frames the *next* reader would otherwise find in front
        of its own: six unread ones sit 36 bytes ahead of the following
        ``SYNC_READ`` stream, so a healthy arm reads back as one joint and five
        servos that did not answer.

        The reply's error byte is not graded: a servo raising a flag still
        answered and still took the write, and what is being distinguished here
        is silence. A mute servo costs one read window, which is what measuring
        silence costs; the settle is paid per servo because the writes are per
        servo, and a torque sweep is a one-shot verb rather than the 30 Hz state
        path :meth:`sync_read` keeps one settle for.

        Every motor is attempted even after one fails: a release that gave up
        part-way would report the arm safe while some joints are still driven.

        Args:
            enabled: ``True`` to energize, ``False`` to release.

        Returns:
            Names of motors that did not acknowledge the write; empty when all
            six answered. A non-empty list after ``enabled=False`` means the arm
            is NOT fully de-energized.

        Raises:
            RuntimeError: When the bus is not open.
        """
        conn = self._require_open("setting torque")
        failed: list[str] = []
        for name, spec in self.motors.items():
            packet = write_packet(spec.motor_id, Register.TORQUE_ENABLE, bytes([1 if enabled else 0]))
            try:
                conn.write(packet)
                time.sleep(_REPLY_SETTLE_S)
                raw = bytes(conn.read(STATUS_OVERHEAD))
                echoed = int(getattr(conn, "in_waiting", 0) or 0)
                if echoed:
                    raw += bytes(conn.read(echoed))
            except OSError as e:
                logger.error("failed to set torque on %s (id %d): %s", name, spec.motor_id, e)
                failed.append(name)
                continue
            # The stream framer rather than a lone packet parse, for the reason
            # :meth:`_sync_read_once` uses it: it skips the host's own echo,
            # which `parse_status_packet` refuses as bytes in front of a frame.
            if spec.motor_id not in parse_sync_read_replies(raw, [spec.motor_id], _ACK_PARAM_COUNT):
                logger.error(
                    "no verified torque ack from %s (id %d); discarding %s",
                    name,
                    spec.motor_id,
                    raw.hex(" ") if raw else "an empty read",
                )
                failed.append(name)
        return failed
