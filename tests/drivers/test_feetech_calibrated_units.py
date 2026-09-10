# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An SO arm's degrees are measured against the travel its calibration recorded.

``lerobot-calibrate`` measures where each joint of one particular arm stops and
writes those counts to a JSON file. Two things then have to agree about what a
degree means: ``lerobot.motors.MotorsBus`` and
:class:`~strands_robots.drivers.feetech.bus.FeetechBus`. They drive the same six
servos, they are pointed at the same file, and a leader arm read by one commonly
commands a follower written by the other.

So the tables below are graded twice. Their expected values are literals
captured from ``_normalize`` / ``_unnormalize``, which pins the conversion on
any box; :class:`TestLeRobotStillAgrees` re-derives the same rows from lerobot
itself, so drift on either side is a failure rather than a discovery months
later on an arm.

``RECORDS`` is the calibration of a physical SO-101 follower, carried here
verbatim rather than read from a host path, so the numbers are an arm's and not
a fixture's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers.feetech.bus import (
    SO_ARM_MOTORS,
    FeetechBus,
    MotorCalibration,
    MotorSpec,
    full_travel_calibration,
    lerobot_calibration_path,
    load_calibration,
)
from strands_robots.drivers.feetech.driver import FeetechDriver
from tests.drivers.conftest import FakeServoPort

#: One physical SO-101 follower, as ``lerobot-calibrate`` measured it.
RECORDS: dict[str, dict[str, int]] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -241, "range_min": 1279, "range_max": 2892},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 296, "range_min": 890, "range_max": 3201},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -551, "range_min": 810, "range_max": 3017},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -536, "range_min": 922, "range_max": 3237},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -1524, "range_min": 163, "range_max": 4032},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -1686, "range_min": 2014, "range_max": 3528},
}

#: ``(joint, counts, value)`` - what a servo reported, and the number lerobot
#: reports for it. The three counts per joint are its two measured stops and the
#: middle of the encoder, which is where the uncalibrated map used to put zero.
READS: list[tuple[str, int, float]] = [
    ("shoulder_pan", 1279, -70.9010989010989),
    ("shoulder_pan", 2048, -3.2967032967032965),
    ("shoulder_pan", 2892, 70.9010989010989),
    ("shoulder_lift", 890, -101.58241758241758),
    ("shoulder_lift", 2048, 0.21978021978021978),
    ("shoulder_lift", 3201, 101.58241758241758),
    ("elbow_flex", 810, -97.01098901098901),
    ("elbow_flex", 2048, 11.824175824175825),
    ("elbow_flex", 3017, 97.01098901098901),
    ("wrist_flex", 922, -101.75824175824175),
    ("wrist_flex", 2048, -2.769230769230769),
    ("wrist_flex", 3237, 101.75824175824175),
    ("wrist_roll", 163, -170.06593406593407),
    ("wrist_roll", 2048, -4.351648351648351),
    ("wrist_roll", 4032, 170.06593406593407),
    ("gripper", 2014, 0.0),
    ("gripper", 2048, 2.2457067371202113),
    ("gripper", 3528, 100.0),
]

#: ``(joint, value, counts)`` - a caller's target and the count lerobot sends.
WRITES: list[tuple[str, float, int]] = [
    ("shoulder_pan", 0.0, 2085),
    ("shoulder_pan", 30.0, 2426),
    ("shoulder_pan", -30.0, 1744),
    ("shoulder_lift", 0.0, 2045),
    ("shoulder_lift", 30.0, 2386),
    ("shoulder_lift", -30.0, 1704),
    ("elbow_flex", 0.0, 1913),
    ("elbow_flex", 30.0, 2254),
    ("elbow_flex", -30.0, 1572),
    ("wrist_flex", 0.0, 2079),
    ("wrist_flex", 30.0, 2420),
    ("wrist_flex", -30.0, 1738),
    ("wrist_roll", 0.0, 2097),
    ("wrist_roll", 30.0, 2438),
    ("wrist_roll", -30.0, 1756),
    ("gripper", 0.0, 2014),
    ("gripper", 30.0, 2468),
    ("gripper", 100.0, 3528),
]


def _records() -> dict[str, MotorCalibration]:
    """The arm above, as the records a bus takes."""
    return {name: MotorCalibration(**fields) for name, fields in RECORDS.items()}


def _calibrated_bus() -> FeetechBus:
    """A bus on that arm. No port is opened - conversion needs no wire."""
    return FeetechBus(port="/dev/fake", calibration=_records())


class TestCalibratedUnits:
    """The arm's own travel decides what a degree and a percent mean."""

    @pytest.mark.parametrize(("joint", "counts", "value"), READS)
    def test_a_reported_count_reads_as_lerobot_reads_it(self, joint: str, counts: int, value: float) -> None:
        assert _calibrated_bus().to_value(joint, counts) == pytest.approx(value, abs=1e-9)

    @pytest.mark.parametrize(("joint", "value", "counts"), WRITES)
    def test_a_target_encodes_to_the_count_lerobot_sends(self, joint: str, value: float, counts: int) -> None:
        assert _calibrated_bus().to_counts(joint, value) == counts

    @pytest.mark.parametrize("joint", sorted(RECORDS))
    def test_a_count_inside_the_measured_travel_round_trips(self, joint: str) -> None:
        """Within one count, which is the encoder's own resolution."""
        bus = _calibrated_bus()
        record = bus.calibration[joint]
        for counts in (record.range_min, (record.range_min + record.range_max) // 2, record.range_max):
            assert bus.to_counts(joint, bus.to_value(joint, counts)) == pytest.approx(counts, abs=1)

    def test_an_uncalibrated_bus_spans_the_servos_own_rotation(self) -> None:
        """A statement about the encoder, not a guess about the arm.

        Every joint reads the same full turn, because that is what an
        uncalibrated servo can say: a fixed per-joint range would claim to know
        where this arm's stops are without having measured them.
        """
        bus = FeetechBus(port="/dev/fake")
        for joint, spec in SO_ARM_MOTORS.items():
            low, high = (0.0, 100.0) if spec.norm_mode == "range_0_100" else (-180.0, 180.0)
            assert bus.to_value(joint, 0) == pytest.approx(low, abs=0.05)
            assert bus.to_value(joint, spec.resolution) == pytest.approx(high, abs=0.05)

    def test_a_target_the_encoder_cannot_hold_is_refused_not_clamped(self) -> None:
        """The one deliberate divergence from ``_unnormalize``, which bounds.

        ``+120`` degrees is past this shoulder's measured travel AND past the
        encoder; lerobot would send a count the servo cannot latch.
        """
        bus = _calibrated_bus()
        with pytest.raises(ValueError, match="outside the travel the encoder can hold"):
            bus.to_counts("shoulder_pan", 200.0)
        with pytest.raises(ValueError, match="outside 0..100 percent"):
            bus.to_counts("gripper", 120.0)

    def test_a_motor_this_bus_does_not_carry_is_named_not_guessed(self) -> None:
        with pytest.raises(ValueError, match="unknown motor 'elbow'"):
            _calibrated_bus().to_value("elbow", 2048)


class TestLeRobotStillAgrees:
    """The literal tables above, re-derived from lerobot itself.

    This is the drift alarm: lerobot changing what ``DEGREES`` means, or this
    package changing its arithmetic, breaks here rather than on an arm.
    """

    @staticmethod
    def _vendor_bus() -> Any:
        pytest.importorskip("lerobot", reason="the vendor cross-check needs lerobot installed")
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors import MotorCalibration as VendorCalibration
        from lerobot.motors.feetech import FeetechMotorsBus

        calibration = {name: VendorCalibration(**fields) for name, fields in RECORDS.items()}
        motors = {
            name: Motor(
                fields["id"],
                "sts3215",
                MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
            )
            for name, fields in RECORDS.items()
        }
        return FeetechMotorsBus(port="/dev/fake", motors=motors, calibration=calibration)

    def test_every_read_row_is_what_normalize_computes(self) -> None:
        vendor = self._vendor_bus()
        derived = [
            (joint, counts, vendor._normalize({RECORDS[joint]["id"]: counts})[RECORDS[joint]["id"]])
            for joint, counts, _ in READS
        ]
        assert derived == READS

    def test_every_write_row_is_what_unnormalize_computes(self) -> None:
        vendor = self._vendor_bus()
        derived = [
            (joint, value, vendor._unnormalize({RECORDS[joint]["id"]: value})[RECORDS[joint]["id"]])
            for joint, value, _ in WRITES
        ]
        assert derived == WRITES


class TestConventionsCopiedDeliberately:
    """Two lerobot conventions that look like bugs until they are pinned."""

    def test_homing_offset_is_the_servos_to_apply_not_this_hosts(self) -> None:
        """The servo already applies it to every count it reports.

        ``lerobot-calibrate`` writes the offset to the servo's own
        ``Homing_Offset`` register, and ``_normalize`` never reads the field.
        A host that subtracted it here would count it twice - the joint would
        read correct at the middle of its travel and wrong everywhere else.
        """
        shifted = {name: MotorCalibration(**{**fields, "homing_offset": 999}) for name, fields in RECORDS.items()}
        moved = FeetechBus(port="/dev/fake", calibration=shifted)
        for joint, counts, value in READS:
            assert moved.to_value(joint, counts) == pytest.approx(value, abs=1e-9)

    def test_drive_mode_reverses_percent_and_leaves_degrees_alone(self) -> None:
        """Where lerobot applies it, and only there.

        Its ``DEGREES`` branch reads the raw count, so a host that inverted
        degrees for a reversed joint would disagree with the arm driving it.
        """
        reversed_records = {name: MotorCalibration(**{**fields, "drive_mode": 1}) for name, fields in RECORDS.items()}
        bus = FeetechBus(port="/dev/fake", calibration=reversed_records)
        assert bus.to_value("gripper", 2014) == pytest.approx(100.0)
        assert bus.to_counts("gripper", 0.0) == 3528
        assert bus.to_value("elbow_flex", 2048) == pytest.approx(11.824175824175825, abs=1e-9)


class TestARecordThatDoesNotDescribeThisArm:
    """A calibration that cannot be trusted is refused at construction."""

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda r: r.pop("wrist_roll"), "no calibration for 'wrist_roll'"),
            (
                lambda r: r.__setitem__("elbow_flex", MotorCalibration(id=9, range_min=0, range_max=4095)),
                "different arm",
            ),
            (
                lambda r: r.__setitem__("gripper", MotorCalibration(id=6, range_min=2014, range_max=2014)),
                "spans no travel",
            ),
        ],
    )
    def test_a_record_that_cannot_be_trusted_is_refused(self, mutate: Any, match: str) -> None:
        records = _records()
        mutate(records)
        with pytest.raises(ValueError, match=match):
            FeetechBus(port="/dev/fake", calibration=records)

    def test_a_record_for_a_motor_this_bus_does_not_carry_is_ignored(self) -> None:
        """A file describes a whole arm; ``motor_ids`` can narrow a bus to part."""
        bus = FeetechBus(port="/dev/fake", motors={"gripper": SO_ARM_MOTORS["gripper"]}, calibration=_records())
        assert set(bus.calibration) == {"gripper"}

    def test_full_travel_is_what_no_calibration_means(self) -> None:
        assert full_travel_calibration(SO_ARM_MOTORS)["wrist_roll"] == MotorCalibration(
            id=5, range_min=0, range_max=4095
        )


class TestLoadingWhatTheCliWrote:
    """The file ``lerobot-calibrate`` produces, read as it was written."""

    def test_the_file_lerobot_writes_loads_field_for_field(self, tmp_path: Path) -> None:
        path = tmp_path / "orange_follower.json"
        path.write_text(json.dumps(RECORDS, indent=4), encoding="utf-8")
        assert load_calibration(path) == _records()

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("not json at all", "not JSON"),
            ('["shoulder_pan"]', "expected an object"),
            ('{"shoulder_pan": 3}', "not a record"),
            ('{"shoulder_pan": {"id": 1, "drive_mode": 0, "range_min": 0, "range_max": 4095}}', "missing"),
            (
                '{"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0,'
                ' "range_max": 4095, "backlash": 3}}',
                "which a record does not",
            ),
            (
                '{"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095.5}}',
                "must be an integer count",
            ),
        ],
    )
    def test_a_file_that_is_not_a_calibration_is_refused(self, tmp_path: Path, content: str, match: str) -> None:
        """Refused rather than completed from defaults: a record with a field
        filled in reports degrees measured against a travel nobody measured."""
        path = tmp_path / "broken.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_calibration(path)

    def test_the_path_is_the_one_lerobot_calibrate_wrote_to(self, tmp_path: Path, monkeypatch: Any) -> None:
        pytest.importorskip("lerobot", reason="the path comes from lerobot's own constants")
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
        constants = pytest.importorskip("lerobot.utils.constants")
        monkeypatch.setattr(constants, "HF_LEROBOT_CALIBRATION", tmp_path)
        assert (
            lerobot_calibration_path("so101_follower", "orange")
            == tmp_path / "robots" / "so101_follower" / "orange.json"
        )

    @pytest.mark.parametrize(
        ("robot_type", "robot_id"),
        [("../../etc", "orange"), ("so101_follower", "../../../secrets"), ("so101_follower", "a/b"), ("", "orange")],
    )
    def test_a_name_that_leaves_the_calibration_directory_is_refused(self, robot_type: str, robot_id: str) -> None:
        with pytest.raises(ValueError, match="one path segment"):
            lerobot_calibration_path(robot_type, robot_id)


class TestTheDriverReachesIt:
    """The fix is reachable from the driver a caller actually builds."""

    def test_a_calibrated_driver_commands_the_counts_lerobot_would(self, tmp_path: Path) -> None:
        """End to end: the file goes in, the SYNC_WRITE frame carries the arm's counts."""
        path = tmp_path / "orange_follower.json"
        path.write_text(json.dumps(RECORDS), encoding="utf-8")
        driver = FeetechDriver(tool_name="so101", port="/dev/fake", calibration=path)
        port = FakeServoPort()
        driver.bus._conn = port
        driver.bus.write_goal_positions({"shoulder_pan": 0.0, "gripper": 0.0})
        payload = port.writes[-1][7:-1]
        pairs = {payload[i]: payload[i + 1] | (payload[i + 2] << 8) for i in range(0, len(payload), 3)}
        assert pairs == {1: 2085, 6: 2014}

    def test_a_calibrated_driver_reads_the_arms_own_degrees(self, tmp_path: Path) -> None:
        path = tmp_path / "orange_follower.json"
        path.write_text(json.dumps(RECORDS), encoding="utf-8")
        driver = FeetechDriver(tool_name="so101", port="/dev/fake", calibration=path)
        driver.bus._conn = FakeServoPort({fields["id"]: 2048 for fields in RECORDS.values()})
        reading = driver.bus.sync_read("Present_Position")
        assert reading["elbow_flex"] == pytest.approx(11.824175824175825, abs=1e-9)
        assert reading["gripper"] == pytest.approx(2.2457067371202113, abs=1e-9)

    def test_status_names_which_travel_the_degrees_are_measured_against(self, tmp_path: Path) -> None:
        """``None`` reads as the keyword the caller forgot, not as a wrong number."""
        path = tmp_path / "orange_follower.json"
        path.write_text(json.dumps(RECORDS), encoding="utf-8")
        calibrated = FeetechDriver(tool_name="so101", port="/dev/fake", calibration=path)
        status = asyncio.run(calibrated.get_status())
        assert status["content"][0]["json"]["calibration_source"] == str(path)
        bare = asyncio.run(FeetechDriver(tool_name="so101", port="/dev/fake").get_status())
        assert bare["content"][0]["json"]["calibration_source"] is None

    def test_a_calibration_that_is_neither_a_path_nor_records_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a path to the JSON"):
            FeetechDriver(tool_name="so101", port="/dev/fake", calibration=7)

    def test_narrowing_the_bus_still_calibrates_what_is_left(self, tmp_path: Path) -> None:
        path = tmp_path / "orange_follower.json"
        path.write_text(json.dumps(RECORDS), encoding="utf-8")
        driver = FeetechDriver(tool_name="so101", port="/dev/fake", calibration=path, motor_ids=(3,))
        assert driver.bus.calibration == {"elbow_flex": MotorCalibration(**RECORDS["elbow_flex"])}
        assert driver.bus.motors == {"elbow_flex": MotorSpec(3)}
