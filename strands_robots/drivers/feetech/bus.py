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
100 open) - the same six IDs in the same wire order, in the same two units,
that lerobot's ``SOFollower`` drives.

*Which counts* those units mean is a property of the individual arm and not of
the model, so it is read from that arm's calibration: the record
``lerobot-calibrate`` measured and :func:`load_calibration` loads. Degrees run
from the middle of the calibrated travel and percent spans it end to end,
exactly as ``lerobot.motors.MotorsBus._normalize`` does for its ``DEGREES``
and ``RANGE_0_100`` modes - the two an SO arm uses. A bus given no calibration
falls back to :func:`full_travel_calibration`, which is a statement about the
encoder rather than a guess about the arm.

A target the encoder cannot hold is **refused, not clamped**. A clamp turns a
caller's 400-degree command into a 180-degree motion and reports success, so
the caller learns nothing and the arm goes somewhere it was not told to go.
That is the one deliberate divergence from ``_unnormalize``, which bounds.

Nothing here imports :mod:`serial` at module load: the import happens in
:meth:`FeetechBus.connect`, so the package stays importable - and its codec
gradeable - on a box with no serial stack at all.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

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


#: A bare path segment: no separator, no ``..``. Applied to the two names
#: :func:`lerobot_calibration_path` interpolates, both of which reach it from a
#: tool call.
_PATH_SEGMENT: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: Fields ``lerobot-calibrate`` writes for every motor, all of them required.
#:
#: Mirrors ``lerobot.motors.MotorCalibration``, which declares them without
#: defaults: a record missing ``range_max`` is not a calibration with one field
#: left to fill in, it is a file that cannot say where the joint stops.
_CALIBRATION_FIELDS: Final[tuple[str, ...]] = ("id", "drive_mode", "homing_offset", "range_min", "range_max")

#: How a motor's counts become the number a caller reads, spelled the way
#: ``lerobot.motors.MotorNormMode`` spells it so the two tables read side by
#: side. An SO arm uses both: five joints in degrees, the gripper in percent.
NormMode = Literal["degrees", "range_0_100"]

#: The two accepted norm modes, stated once so the :meth:`MotorSpec.__post_init__`
#: guard and the ``to_value`` / ``to_counts`` dispatch agree.
_NORM_MODES: Final[frozenset[str]] = frozenset({"degrees", "range_0_100"})


@dataclass(frozen=True, slots=True)
class MotorCalibration:
    """One motor's measured travel, as ``lerobot-calibrate`` recorded it.

    Field names and order mirror ``lerobot.motors.MotorCalibration``, so an
    entry of the JSON that CLI writes constructs one directly - which is what
    :func:`load_calibration` does.

    Attributes:
        id: The servo's ID on the bus. Graded against the ID
            :data:`SO_ARM_MOTORS` gives the joint of the same name, because a
            record that disagrees would calibrate one joint with another
            joint's travel.
        drive_mode: ``1`` when the joint's positive direction is reversed.
            Applied to percent only, which is where lerobot applies it: its
            ``DEGREES`` branch reads the raw count, so a host that inverted
            degrees here would disagree with the arm that drives it.
        homing_offset: The offset ``lerobot-calibrate`` wrote to the servo's own
            ``Homing_Offset`` register. Carried so a caller can see it, and
            deliberately **not** applied by this host: the servo already applies
            it to every count it reports, so subtracting it again here would
            count it twice. ``_normalize`` does not read this field either.
        range_min: Lowest count of the measured travel.
        range_max: Highest count of the measured travel.
    """

    id: int
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = MAX_GOAL_POSITION


@dataclass(frozen=True, slots=True)
class MotorSpec:
    """One servo on the bus: its ID, the unit it speaks, and its full scale.

    The per-arm half of a joint - where it stops and how far it travels - lives
    in :class:`MotorCalibration`. This is the half that is a property of the
    servo model and the joint's role, and so is the same on every SO arm.

    Attributes:
        motor_id: The servo's ID on the shared half-duplex bus.
        norm_mode: ``"degrees"`` for a joint, ``"range_0_100"`` for a gripper.
        resolution: Highest count the encoder reports, i.e. lerobot's
            ``MODEL_RESOLUTION`` entry minus one. Defaults to
            :data:`~strands_robots.drivers.feetech.protocol.MAX_GOAL_POSITION`,
            the STS/SMS full scale - an SCS-series servo spans a quarter of it
            and must be given its own.
    """

    motor_id: int
    norm_mode: NormMode = "degrees"
    resolution: int = MAX_GOAL_POSITION

    def __post_init__(self) -> None:
        if self.norm_mode not in _NORM_MODES:
            raise ValueError(f"MotorSpec: norm_mode must be one of {sorted(_NORM_MODES)}, got {self.norm_mode!r}")


#: The six servos of an SO-100 / SO-101 follower, in wire order.
#:
#: The IDs and the gripper's percent domain are the ones lerobot's
#: ``SOFollower`` declares (``Motor(1, "sts3215", DEGREES)`` through
#: ``Motor(6, "sts3215", RANGE_0_100)``) and the ones
#: :mod:`strands_robots.tools.pose_tool` drives on the physical arm. Keeping one
#: map for the tool and the driver is what stops the two commanding different
#: joints by the same name.
#:
#: What is deliberately absent is a per-joint degree range. An SO arm's
#: reachable span is measured per arm, and lives in :class:`MotorCalibration`.
SO_ARM_MOTORS: Final[dict[str, MotorSpec]] = {
    "shoulder_pan": MotorSpec(1),
    "shoulder_lift": MotorSpec(2),
    "elbow_flex": MotorSpec(3),
    "wrist_flex": MotorSpec(4),
    "wrist_roll": MotorSpec(5),
    "gripper": MotorSpec(6, "range_0_100"),
}


def full_travel_calibration(motors: Mapping[str, MotorSpec]) -> dict[str, MotorCalibration]:
    """Calibrate ``motors`` to the servo's own full travel.

    The fallback for a bus given no calibration. It says something true about
    the encoder - count ``0`` is one end of the servo's rotation and
    ``resolution`` the other - and is therefore wrong about the *arm* by
    however far its stops sit inside that rotation, which is the thing
    calibrating measures. It is not a guess about the arm.

    Args:
        motors: The bus's motors, by name.

    Returns:
        Motor name -> a record spanning ``0..spec.resolution``.
    """
    return {
        name: MotorCalibration(id=spec.motor_id, range_min=0, range_max=spec.resolution)
        for name, spec in motors.items()
    }


def load_calibration(path: str | Path) -> dict[str, MotorCalibration]:
    """Read the calibration ``lerobot-calibrate`` wrote for one arm.

    Args:
        path: The JSON file, e.g. what :func:`lerobot_calibration_path` returns.

    Returns:
        Motor name -> :class:`MotorCalibration`, every field read from the file.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not a JSON object of complete records with
            integer counts. Refused rather than completed from defaults,
            because a record with a field filled in reports degrees measured
            against a travel nobody measured.
    """
    location = Path(path).expanduser()
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"calibration {location}: not JSON ({e})") from e
    if not isinstance(raw, dict):
        raise ValueError(
            f"calibration {location}: expected an object of motor name -> record, got {type(raw).__name__}"
        )
    loaded: dict[str, MotorCalibration] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"calibration {location}: {name} is a {type(entry).__name__}, not a record")
        if missing := [field for field in _CALIBRATION_FIELDS if field not in entry]:
            raise ValueError(
                f"calibration {location}: {name} is missing {missing}; every record carries {list(_CALIBRATION_FIELDS)}"
            )
        if unknown := sorted(set(entry) - set(_CALIBRATION_FIELDS)):
            raise ValueError(
                f"calibration {location}: {name} carries {unknown}, which a record does not; "
                f"a record is {list(_CALIBRATION_FIELDS)}"
            )
        for field in _CALIBRATION_FIELDS:
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"calibration {location}: {name}.{field} must be an integer count, "
                    f"got {type(value).__name__}: {value!r}"
                )
        loaded[name] = MotorCalibration(**{field: int(entry[field]) for field in _CALIBRATION_FIELDS})
    return loaded


def lerobot_calibration_path(robot_type: str, robot_id: str) -> Path:
    """Locate one arm's calibration inside lerobot's own calibration directory.

    Both the directory and the layout under it are read from
    ``lerobot.utils.constants`` rather than spelled here, so a caller who moved
    it with ``HF_LEROBOT_CALIBRATION`` gets the file they calibrated and this
    package cannot drift from the CLI that writes it.

    Args:
        robot_type: The robot lerobot calibrated, e.g. ``"so101_follower"``.
        robot_id: The ``--robot.id`` that run was given, e.g. ``"my_arm"``.

    Returns:
        The path, whether or not it exists - a caller that wants the servo's
        full travel as a fallback tests :meth:`~pathlib.Path.is_file` itself.

    Raises:
        ImportError: lerobot is not installed.
        ValueError: Either name is not a bare path segment. A separator or a
            ``..`` here would read a file from outside the calibration
            directory, and both names reach this from a tool call.
    """
    for label, value in (("robot_type", robot_type), ("robot_id", robot_id)):
        if not isinstance(value, str) or not _PATH_SEGMENT.fullmatch(value):
            raise ValueError(
                f"lerobot_calibration_path: {label} must match {_PATH_SEGMENT.pattern} "
                f"(one path segment, no separator), got {value!r}"
            )
    constants: Any = require_optional(
        "lerobot.utils.constants",
        pip_install="lerobot",
        extra="lerobot",
        purpose="the calibration directory lerobot-calibrate writes to",
    )
    # str() around both: the module is typed Any (it is optional), and a Path
    # built from an Any is an Any that no longer grades this function's return.
    directory = Path(str(constants.HF_LEROBOT_CALIBRATION))
    return directory / str(constants.ROBOTS) / robot_type / f"{robot_id}.json"


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
        calibration: What each of those servos measured when
            ``lerobot-calibrate`` last ran on THIS arm, from
            :func:`load_calibration`. ``None`` falls back to
            :func:`full_travel_calibration` - correct about the servo, and
            wrong about the arm by however far its stops sit inside the
            servo's rotation.
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
        ValueError: ``baud_rate`` is not a positive integer, ``timeout`` is not
            a positive finite number, or ``calibration`` does not describe
            every motor on this bus.
    """

    def __init__(
        self,
        port: str | None,
        baud_rate: int = 1_000_000,
        motors: dict[str, MotorSpec] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        if (reason := positive_count_error(baud_rate, "baud_rate", type(self).__name__)) is not None:
            raise ValueError(reason)
        if (reason := positive_finite_number_error(timeout, "timeout", type(self).__name__)) is not None:
            raise ValueError(reason)
        self.port = port
        self.baud_rate = baud_rate
        self.motors = dict(SO_ARM_MOTORS if motors is None else motors)
        self.timeout = timeout
        self.calibration = self._resolve_calibration(calibration)
        self._conn: Any | None = None

    def _resolve_calibration(self, calibration: dict[str, MotorCalibration] | None) -> dict[str, MotorCalibration]:
        """Pair every motor with its record, or refuse naming what disagrees.

        A record this bus has no motor for is ignored: a calibration file
        describes a whole arm, and ``motor_ids`` can narrow a bus to part of
        one. A motor with NO record is refused rather than given the servo's
        full travel, because that is the one case where a caller who calibrated
        the arm would silently read back the uncalibrated degrees they
        calibrated to correct.

        Args:
            calibration: The caller's records, or ``None`` for full travel.

        Returns:
            One record per motor on this bus.

        Raises:
            ValueError: A motor has no record, a record names a different servo
                ID than the motor of that name, or a record spans no travel.
        """
        if calibration is None:
            return full_travel_calibration(self.motors)
        resolved: dict[str, MotorCalibration] = {}
        for name, spec in self.motors.items():
            record = calibration.get(name)
            if record is None:
                raise ValueError(
                    f"FeetechBus: no calibration for {name!r}; the records cover {sorted(calibration)} "
                    f"and this bus carries {sorted(self.motors)}"
                )
            if record.id != spec.motor_id:
                raise ValueError(
                    f"FeetechBus: the calibration for {name!r} measured servo id {record.id}, but this "
                    f"bus drives id {spec.motor_id} by that name - one of the two is a different arm"
                )
            if record.range_min >= record.range_max:
                raise ValueError(
                    f"FeetechBus: the calibration for {name!r} spans no travel "
                    f"(range_min={record.range_min}, range_max={record.range_max}); a joint whose "
                    "positive direction is reversed says so with drive_mode, not by swapping the two"
                )
            resolved[name] = record
        return resolved

    # ------------------------------------------------------------------ #
    # Units.                                                              #
    # ------------------------------------------------------------------ #

    def _calibrated(self, name: str) -> tuple[MotorSpec, MotorCalibration]:
        """Return one motor's spec and record, or raise naming the bus's motors."""
        spec = self.motors.get(name)
        if spec is None:
            raise ValueError(f"FeetechBus: unknown motor {name!r}; this bus carries {sorted(self.motors)}")
        return spec, self.calibration[name]

    def to_value(self, name: str, counts: int) -> float:
        """Decode one motor's encoder counts into the unit a caller reads.

        Cell for cell what ``lerobot.motors.MotorsBus._normalize`` computes for
        this motor's norm mode, so an arm read through this bus and through
        ``SOFollower`` reports the same number.

        Args:
            name: A motor on this bus.
            counts: The count the servo reported.

        Returns:
            Degrees from the middle of the calibrated travel, or percent open
            across it. Degrees are reported for whatever count arrived, including
            one outside the measured travel - that is where the joint actually
            is, and discarding it would discard the arm's own state. Percent is
            bounded to ``0..100``, because ``_normalize`` bounds it and a gripper
            at 103 percent open is not a thing a caller can act on.

        Raises:
            ValueError: ``name`` is not on this bus.
        """
        spec, record = self._calibrated(name)
        if spec.norm_mode == "degrees":
            middle = (record.range_min + record.range_max) / 2
            return (counts - middle) * 360.0 / spec.resolution
        bounded = min(record.range_max, max(record.range_min, counts))
        percent = (bounded - record.range_min) / (record.range_max - record.range_min) * 100.0
        return 100.0 - percent if record.drive_mode else percent

    def to_counts(self, name: str, value: float) -> int:
        """Encode a target in the motor's own unit as encoder counts.

        The inverse of :meth:`to_value`, and cell for cell
        ``_unnormalize`` - except that a target the encoder cannot hold is
        refused instead of bounded, for the reason the module docstring gives.

        Args:
            name: A motor on this bus.
            value: The target, in degrees or percent open.

        Returns:
            A count in ``0..spec.resolution``.

        Raises:
            ValueError: ``name`` is not on this bus, a percent target is
                outside ``0..100``, or the target maps outside the encoder.
        """
        spec, record = self._calibrated(name)
        if spec.norm_mode == "degrees":
            middle = (record.range_min + record.range_max) / 2
            counts = int(value * spec.resolution / 360.0 + middle)
        else:
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"FeetechBus: {name} target {value} is outside 0..100 percent open")
            percent = 100.0 - value if record.drive_mode else value
            counts = int(percent / 100.0 * (record.range_max - record.range_min) + record.range_min)
        if not 0 <= counts <= spec.resolution:
            raise ValueError(
                f"FeetechBus: {name} target {value} is outside the travel the encoder can hold: "
                f"it maps to {counts} counts, and the servo reports 0..{spec.resolution}"
            )
        return counts

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
            unit - :meth:`to_value` against this arm's calibration - and the
            other registers are raw signed counts. Motors that did not answer
            are absent.

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
            out[name] = self.to_value(name, value) if register == "Present_Position" else float(value)
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
            spec, _record = self._calibrated(name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"FeetechBus: {name} target must be a finite number, got {type(value).__name__}: {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"FeetechBus: {name} target must be finite, got {value!r}")
            counts = self.to_counts(name, number)
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
