"""The teleop safety envelope must bound the unit the frame is actually in.

Q25: ``DEFAULT_INPUT_VALUE_ABS`` is 4*pi - a RADIAN bound - but lerobot
normalises an SO-101's Feetech positions to DEGREES (``wrist_roll`` rests at
170) with the gripper in percent. Measured on the real arms before this fix:
209 frames published, 176 rejected, 0 applied, and the only trace was one line
in the follower's own log.

Two rules these tests exist to keep:
  * the unit comes from the RECEIVING robot's own declaration, never from the
    frame - a sender must not choose the bound that constrains it;
  * an unknown unit keeps the radian default instead of the widest row.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.mesh.input import InputReceiver, declared_units
from strands_robots.mesh.security import (
    DEFAULT_INPUT_VALUE_ABS,
    INPUT_ENVELOPES_BY_UNIT,
    ValidationError,
    input_envelope_for_units,
    input_frame_slew_violation,
    validate_input_frame,
)


class _Mode:
    def __init__(self, name: str) -> None:
        self.name = name


class _Motor:
    def __init__(self, name: str) -> None:
        self.norm_mode = _Mode(name)


class _Bus:
    def __init__(self, **motors: str) -> None:
        self.motors = {k: _Motor(v) for k, v in motors.items()}


class _LeRobot:
    """Shape of lerobot's SOFollower: the motor table hangs off ``.bus``."""

    def __init__(self, **motors: str) -> None:
        self.bus = _Bus(**motors)


class _Wrapper:
    """Shape of a strands Robot wrapping a hardware robot wrapping lerobot."""

    def __init__(self, inner: object) -> None:
        self.robot = inner


SO101 = {
    "shoulder_pan": "DEGREES",
    "shoulder_lift": "DEGREES",
    "elbow_flex": "DEGREES",
    "wrist_flex": "DEGREES",
    "wrist_roll": "DEGREES",
    "gripper": "RANGE_0_100",
}


# ---------------------------------------------------------------- the bug itself


def test_a_real_so101_frame_is_refused_without_a_declaration_and_accepted_with_one():
    """The exact frame the arms send: wrist_roll at 170, gripper in percent."""
    frame = {"wrist_roll.pos": 170.0, "shoulder_lift.pos": -46.0, "gripper.pos": 2.3}

    # Before: the radian bound refuses it, which is what "0 applied" looked like.
    with pytest.raises(ValidationError) as err:
        validate_input_frame(frame)
    assert "out of range" in str(err.value)

    # After: the receiver's own robot says these joints are degrees.
    units = declared_units(_Wrapper(_LeRobot(**SO101)))
    value_abs, _slew, note = input_envelope_for_units(units)
    assert validate_input_frame(frame, value_abs_by_key=value_abs) == frame
    assert "5 joints in deg" in note and "1 joint in pct" in note


def test_the_percent_gripper_keeps_its_own_bound_in_the_same_frame():
    """A degree shoulder must not widen the percent gripper beside it."""
    units = declared_units(_LeRobot(**SO101))
    value_abs, _slew, _note = input_envelope_for_units(units)
    assert value_abs["shoulder_pan.pos"] == INPUT_ENVELOPES_BY_UNIT["deg"][0]
    assert value_abs["gripper.pos"] == INPUT_ENVELOPES_BY_UNIT["pct"][0]
    # 300 is a plausible degree value and an impossible percent one.
    validate_input_frame({"shoulder_pan.pos": 300.0}, value_abs_by_key=value_abs)
    with pytest.raises(ValidationError):
        validate_input_frame({"gripper.pos": 300.0}, value_abs_by_key=value_abs)


# ---------------------------------------------------------------- the rules


def test_an_unknown_unit_keeps_the_radian_default_rather_than_the_widest_row():
    value_abs, slew, note = input_envelope_for_units({"j.pos": "furlongs"})
    assert value_abs == {} and slew == {}
    assert "unrecognised" in note
    with pytest.raises(ValidationError):
        validate_input_frame({"j.pos": 170.0}, value_abs_by_key=value_abs)


def test_nothing_declared_changes_nothing():
    value_abs, slew, note = input_envelope_for_units({})
    assert (value_abs, slew) == ({}, {})
    assert "radian defaults" in note
    # And the radian default is still exactly what it was.
    assert validate_input_frame({"j.pos": math.pi})["j.pos"] == pytest.approx(math.pi)
    assert DEFAULT_INPUT_VALUE_ABS == pytest.approx(4 * math.pi)


def test_units_are_read_from_hardware_not_from_the_frame():
    """There is no path from frame content to the bound.

    The receiver derives its envelope at start() from its own robot; a frame
    that claims a unit gets no say. Pinned by shape: a frame key that looks
    like a declaration is treated as just another joint, and is bounded.
    """
    units = declared_units(_LeRobot(**SO101))
    value_abs, _slew, _note = input_envelope_for_units(units)
    with pytest.raises(ValidationError):
        validate_input_frame({"units.pos": 5000.0}, value_abs_by_key=value_abs)


def test_a_broken_motor_table_falls_back_instead_of_raising():
    class Hostile:
        @property
        def bus(self):  # noqa: ANN201 - deliberately explodes
            raise RuntimeError("bus is on fire")

    assert declared_units(Hostile()) == {}
    assert declared_units(_LeRobot()) == {}  # empty table, not a crash


# ---------------------------------------------------------------- slew, per joint


def test_slew_is_bounded_per_joint_in_that_joint_s_unit():
    units = declared_units(_LeRobot(**SO101))
    _value_abs, slew, _note = input_envelope_for_units(units)
    # A degree joint travelling 100 deg in 0.1s = 1000 deg/s: over the 720-unit
    # value bound's companion slew (1440)? No - allowed. The percent gripper
    # moving the same 100 units is 1000 %/s, way past its 400 bound.
    previous = {"shoulder_pan.pos": (0.0, 0.0), "gripper.pos": (0.0, 0.0)}
    assert (
        input_frame_slew_violation(
            {"shoulder_pan.pos": 100.0}, previous, 0.1, 0.0, max_slew_by_key=slew
        )
        is None
    )
    reason = input_frame_slew_violation(
        {"gripper.pos": 100.0}, previous, 0.1, 0.0, max_slew_by_key=slew
    )
    assert reason is not None and "gripper.pos" in reason and "400" in reason


def test_the_reported_bound_is_the_one_that_actually_refused():
    """A message naming the wrong bound sends the operator to the wrong knob."""
    previous = {"a.pos": (0.0, 0.0), "b.pos": (0.0, 0.0)}
    reason = input_frame_slew_violation(
        {"a.pos": 50.0, "b.pos": 900.0},
        previous,
        0.1,
        0.0,
        max_slew_by_key={"a.pos": 10.0, "b.pos": 1440.0},
    )
    # b moved furthest but a's bound is the tighter one; the worst offender is
    # by SPEED-over-bound reporting, so whichever is named must carry its bound.
    assert reason is not None
    named = "a.pos" if "a.pos" in reason else "b.pos"
    expected = "10" if named == "a.pos" else "1440"
    assert expected in reason


# ---------------------------------------------------------------- the receiver


def test_receiver_derives_the_envelope_at_start_and_reports_it(monkeypatch):
    class FakeMesh:
        def subscribe(self, topic, callback=None, name=None):  # noqa: ANN001
            return "sub-1"

        def unsubscribe(self, name):  # noqa: ANN001
            return None

    receiver = InputReceiver(FakeMesh(), _Wrapper(_LeRobot(**SO101)), "so101-arm-1", "leader")
    assert receiver.stats["envelope_units"] == {}  # nothing derived before start
    receiver.start()
    try:
        units = receiver.stats["envelope_units"]
        assert units["wrist_roll.pos"] == "deg"
        assert units["gripper.pos"] == "pct"
    finally:
        receiver.stop()
