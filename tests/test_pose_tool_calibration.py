# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``pose_tool`` must drive the arm through its lerobot calibration.

``MotorController.degrees_to_position`` mapped degrees onto ticks by normalising a
HARDCODED per-joint degree range, so 0 degrees became tick 2047 on every joint and
tick 0 on the gripper. Real calibration puts the centres elsewhere. Measured
against ``so_follower/so101.json`` on this machine:

    motor            pose_tool@0deg   lerobot@0deg   error
    shoulder_pan               2047           1895   +152 ticks (13.4 deg)
    elbow_flex                 2047           2008    +39 ticks ( 3.4 deg)
    wrist_flex                 2047           1937   +110 ticks ( 9.7 deg)
    gripper                       0           2029  -2029 ticks (178   deg)

The gripper is the dangerous one: a "fully closed" command drove far past the
mechanical stop. The consequence is that a pose stored or commanded through this
tool referred to a DIFFERENT physical configuration than the same numbers sent
through the policy / teleop path, which is why ``molmoact2_pickplace.py`` documents
this and refuses to use ``pose_tool``.

The conversion now mirrors ``lerobot.motors.MotorsBus._unnormalize`` exactly:
``MotorNormMode.DEGREES`` is mid-point-centred (``val * 4095/360 + mid``, where
``mid = (range_min + range_max)/2``) and the gripper's ``RANGE_0_100`` maps 0..100
across the measured span. No serial port is opened.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.tools.pose_tool import (
    _TICK_SPAN,
    MotorController,
    load_lerobot_calibration,
)

# A real SO-101 follower calibration (values from so_follower/so101.json on the
# dev machine), so the numbers under test are not invented.
_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 445, "range_min": 859, "range_max": 2931},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 254, "range_min": 865, "range_max": 3234},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -287, "range_min": 900, "range_max": 3116},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -375, "range_min": 782, "range_max": 3092},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -1443, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -1796, "range_min": 2029, "range_max": 3517},
}


def _lerobot_tick(motor: str, degrees: float) -> int:
    """The authority: lerobot MotorsBus._unnormalize, transcribed."""
    record = _CALIBRATION[motor]
    range_min, range_max = record["range_min"], record["range_max"]
    if motor == "gripper":  # MotorNormMode.RANGE_0_100
        bounded = min(100.0, max(0.0, degrees))
        return int((bounded / 100.0) * (range_max - range_min) + range_min)
    mid = (range_min + range_max) / 2  # MotorNormMode.DEGREES
    return int(degrees * _TICK_SPAN / 360 + mid)


def _calibrated() -> MotorController:
    return MotorController(port="/dev/null", calibration=_CALIBRATION)


class TestCalibratedConversionMatchesLerobot:
    @pytest.mark.parametrize("motor", sorted(_CALIBRATION))
    def test_zero_degrees_matches(self, motor):
        """Regression: every joint used to land on tick 2047 (gripper on 0)."""
        assert _calibrated().degrees_to_position(motor, 0.0) == _lerobot_tick(motor, 0.0)

    @pytest.mark.parametrize("degrees", [-45.0, -10.0, 0.0, 15.0, 60.0])
    def test_arm_joint_across_the_range(self, degrees):
        controller = _calibrated()
        for motor in ("shoulder_pan", "elbow_flex", "wrist_flex", "wrist_roll"):
            assert controller.degrees_to_position(motor, degrees) == _lerobot_tick(motor, degrees), motor

    @pytest.mark.parametrize("percent", [0.0, 25.0, 50.0, 100.0])
    def test_gripper_uses_range_0_100_not_a_raw_fraction(self, percent):
        """The gripper's span starts at range_min (2029 here), not at tick 0."""
        assert _calibrated().degrees_to_position("gripper", percent) == _lerobot_tick("gripper", percent)

    def test_closed_gripper_is_not_tick_zero(self):
        """The dangerous case: tick 0 is ~178 deg past the mechanical stop."""
        assert _calibrated().degrees_to_position("gripper", 0.0) == 2029


class TestRoundTrip:
    @pytest.mark.parametrize("motor,value", [("shoulder_pan", 30.0), ("wrist_flex", -20.0), ("gripper", 45.0)])
    def test_degrees_to_ticks_and_back(self, motor, value):
        controller = _calibrated()

        ticks = controller.degrees_to_position(motor, value)

        # Tolerance is one tick of quantisation (360/4095 deg, or 100/1488 percent).
        assert controller.position_to_degrees(motor, ticks) == pytest.approx(value, abs=0.15)

    def test_invalid_calibration_span_is_rejected(self):
        """A degenerate range would divide by zero; fail loudly instead."""
        broken = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 5, "range_max": 5}}
        controller = MotorController(port="/dev/null", calibration=broken)

        with pytest.raises(ValueError, match="range_min == range_max"):
            controller.degrees_to_position("shoulder_pan", 0.0)


class TestUncalibratedFallbackIsUnchanged:
    def test_without_calibration_the_legacy_mapping_still_applies(self):
        """No calibration file must not become a crash; it degrades and warns."""
        controller = MotorController(port="/dev/null")

        # int() truncation of 0.5 * 4095 -> 2047, the value the warning quotes.
        assert controller.degrees_to_position("shoulder_pan", 0.0) == _TICK_SPAN // 2
        assert controller.degrees_to_position("gripper", 0.0) == 0

    def test_calibration_changes_the_answer(self):
        """Guards the whole point: the two paths must genuinely differ."""
        uncalibrated = MotorController(port="/dev/null").degrees_to_position("shoulder_pan", 0.0)
        calibrated = _calibrated().degrees_to_position("shoulder_pan", 0.0)

        assert uncalibrated != calibrated
        assert abs(uncalibrated - calibrated) > 100  # ~13 degrees on this arm


class TestCalibrationLoader:
    def test_loads_from_the_lerobot_layout(self, tmp_path, monkeypatch):
        """Directory is the driver CLASS name so_follower, not so101_follower."""
        target = tmp_path / "robots" / "so_follower"
        target.mkdir(parents=True)
        (target / "myarm.json").write_text(json.dumps(_CALIBRATION))
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))

        loaded = load_lerobot_calibration("myarm")

        assert loaded is not None
        assert set(loaded) == set(_CALIBRATION)

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))

        assert load_lerobot_calibration("nope") is None

    def test_malformed_file_returns_none_rather_than_raising(self, tmp_path, monkeypatch):
        target = tmp_path / "robots" / "so_follower"
        target.mkdir(parents=True)
        (target / "bad.json").write_text("{not json")
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))

        assert load_lerobot_calibration("bad") is None

    def test_empty_object_returns_none(self, tmp_path, monkeypatch):
        target = tmp_path / "robots" / "so_follower"
        target.mkdir(parents=True)
        (target / "empty.json").write_text("{}")
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))

        assert load_lerobot_calibration("empty") is None
