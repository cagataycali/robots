# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An interpolated pose move must answer for the joints it actually commanded.

``MotorController._smooth_move`` ended with a literal ``return True`` and threw
away every ``move_motor`` result, so the interpolating path reported success no
matter what reached the bus. Two things it did silently:

* A motor whose current position did not arrive has no start point, so no
  trajectory is built for it and it is never commanded. It was dropped from the
  move without a word.
* A write that raised - a pulled cable, a bus held by another process - was
  logged by ``move_motor`` and then discarded by the loop.

On an identical bus the sibling one flag away told the truth:
``move_multiple_motors(smooth=False)`` reads every ``move_motor`` outcome. So a
single ``smooth`` flag selected between two contracts, and ``smooth`` defaults
to ``True`` - the default path was the lying one. ``reset_to_home`` passes
``smooth=True`` itself and cannot opt out at all.

The tests assert against the BYTES that reach the bus rather than the status
text, for the reason the sibling emergency-stop module gives: the wording was
already right, and only the packets say whether the arm moved.

No real serial port is opened: ``serial.Serial`` is replaced and every call
passes an explicit fake ``port`` -- ``pose_tool``'s ``port`` defaults to
``/dev/ttyACM0``, so a portless call would drive an arm plugged into the machine
running the suite.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Any

import pytest
import serial

from strands_robots.tools.pose_tool import MotorController, pose_tool

from .conftest import FakeSerial, position_packet

_PORT = "/dev/ttyTEST"
_GOAL_POSITION_ADDR = 0x2A
_INST_WRITE = 0x03
# The two joints these tests drive, and the ids pose_tool assigns them.
_TARGET = {"shoulder_pan": 10.0, "elbow_flex": -5.0}
_IDS = {"shoulder_pan": 1, "elbow_flex": 3}
# Enough increments to tell "wrote nothing" from "wrote a partial trajectory".
_STEPS = 3
_DELAY = 1e-6


def _goal_position_ids(writes: list[bytes]) -> list[int]:
    """Motor ids that received a ``Goal_Position`` write, in order."""
    return [
        w[2]
        for w in writes
        if len(w) >= 8 and w[:2] == b"\xff\xff" and w[4] == _INST_WRITE and w[5] == _GOAL_POSITION_ADDR
    ]


def _texts(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result["content"] if "text" in c)


class _SelectiveSerial(FakeSerial):
    """A bus where reads and position writes fail independently, per motor id.

    Both are real half-duplex symptoms: a servo that has dropped off the chain
    answers nothing, and a bus that has gone away raises on write. They are
    separate knobs because the two are separate failures - a joint with no start
    position is never commanded, a joint whose write was refused was commanded
    and did not take it - and a fake that coupled them could not tell the two
    halves of the fix apart. Only the ``Goal_Position`` write is refused, so a
    refused motor still answers its position read.
    """

    answering: set[int] = set()
    refusing: set[int] = set()

    def write(self, data: bytes) -> None:
        packet = bytes(data)
        addressed = packet[2]
        is_position_write = packet[4] == _INST_WRITE and packet[5] == _GOAL_POSITION_ADDR
        if is_position_write and addressed in self.refusing:
            raise serial.SerialException(f"bus write failed for id {addressed}")
        super().write(packet)

    def read(self, n: int = 1) -> bytes:
        asked = self.writes[-1][2] if self.writes else 0x01
        if asked not in self.answering:
            return b""
        return position_packet(motor_id=asked)


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch):
    """A configurable bus. Mutate ``answering`` / ``refusing`` before the call."""
    cfg: dict[str, set[int]] = {"answering": set(range(1, 7)), "refusing": set()}
    made: list[_SelectiveSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _SelectiveSerial:
        fs = _SelectiveSerial(port, baudrate, timeout)
        fs.answering = cfg["answering"]
        fs.refusing = cfg["refusing"]
        made.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    cfg["made"] = made  # type: ignore[assignment]
    return cfg


def _connected(bus: dict[str, Any]) -> MotorController:
    controller = MotorController(_PORT)
    connected, error = controller.connect()
    assert connected, error
    return controller


def _interpolate(bus: dict[str, Any], **kwargs: Any) -> bool:
    """Drive the interpolating branch and return what it reported."""
    controller = _connected(bus)
    return bool(controller.move_multiple_motors(dict(_TARGET), smooth=True, steps=_STEPS, step_delay=_DELAY, **kwargs))


def _one_shot(bus: dict[str, Any]) -> bool:
    """Drive the one-shot branch on the same bus, for comparison."""
    controller = _connected(bus)
    return bool(controller.move_multiple_motors(dict(_TARGET), smooth=False))


class TestTheInterpolatedMoveAnswersForWhatItCommanded:
    """Every joint the caller asked for is either commanded or reported."""

    def test_a_bus_that_answers_no_read_reports_the_move_it_never_made(self, bus: dict[str, Any]) -> None:
        """With no start pose, nothing is interpolated - and nothing is claimed.

        Pre-fix this returned True having written zero Goal_Position packets.
        """
        bus["answering"].clear()

        reported = _interpolate(bus)

        assert reported is False
        assert _goal_position_ids(bus["made"][0].writes) == []

    def test_a_motor_that_did_not_answer_is_reported_not_dropped(self, bus: dict[str, Any]) -> None:
        """One silent joint is left uncommanded, so the move is not a success."""
        bus["answering"].discard(_IDS["elbow_flex"])

        reported = _interpolate(bus)

        commanded = set(_goal_position_ids(bus["made"][0].writes))
        assert reported is False
        assert commanded == {_IDS["shoulder_pan"]}, "the silent joint was commanded after all"

    def test_the_uncommanded_joint_is_named(self, bus: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        """The operator is told which joint will not move, by name."""
        bus["answering"].discard(_IDS["elbow_flex"])

        with caplog.at_level(logging.ERROR, logger="strands_robots.tools.pose_tool"):
            _interpolate(bus)

        errors = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
        assert "elbow_flex" in errors, errors
        assert "shoulder_pan" not in errors, "a joint that was commanded was named as uncommanded"

    def test_a_write_the_bus_refused_is_reported(self, bus: dict[str, Any]) -> None:
        """A joint whose Goal_Position write raised did not reach the target."""
        bus["refusing"].add(_IDS["elbow_flex"])

        reported = _interpolate(bus)

        assert reported is False
        # The read succeeded, so this is the write half of the verdict on its own.
        assert set(_goal_position_ids(bus["made"][0].writes)) == {_IDS["shoulder_pan"]}

    def test_both_branches_now_agree_on_the_same_bus(self, bus: dict[str, Any]) -> None:
        """The ``smooth`` flag selects a trajectory shape, not a contract.

        Pre-fix the interpolating branch answered True on the bus where the
        one-shot branch answered False.
        """
        bus["refusing"].update(_IDS.values())

        assert _one_shot(bus) is False
        assert _interpolate(bus) is False


class TestTheToolReportsIt:
    """The bool reaches the caller: each interpolating action refuses."""

    def test_move_multiple_does_not_report_a_pose_it_never_commanded(self, cwd_tmp: Any, bus: dict[str, Any]) -> None:
        """Pre-fix this listed both target angles under status="success"."""
        bus["answering"].clear()

        result = pose_tool(
            action="move_multiple",
            robot_id="hw_arm",
            port=_PORT,
            positions=dict(_TARGET),
            smooth=True,
            steps=_STEPS,
            step_delay=0.001,
        )

        assert result["status"] == "error", _texts(result)
        assert _goal_position_ids(bus["made"][0].writes) == []

    def test_reset_to_home_cannot_opt_out_of_the_interpolating_path(self, cwd_tmp: Any, bus: dict[str, Any]) -> None:
        """``reset_to_home`` passes ``smooth=True`` itself, so it had no honest branch.

        This is the action an operator reaches for to put the arm somewhere
        known, and it was the one that could not decline to interpolate.
        """
        bus["answering"].clear()

        result = pose_tool(action="reset_to_home", robot_id="hw_arm", port=_PORT, steps=_STEPS, step_delay=0.001)

        assert result["status"] == "error", _texts(result)
        assert _goal_position_ids(bus["made"][0].writes) == []


class TestEveryCommandingMethodAnswersForItself:
    """Derived: no method on the controller may discard a ``move_motor`` result.

    The rule rather than a list of methods, so a fourth commanding path is held
    to it the hour it lands.
    """

    @staticmethod
    def _callers_that_discard() -> list[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(MotorController)))
        discarding: list[str] = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for stmt in ast.walk(fn):
                # A bare expression statement throws the return value away.
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "move_motor":
                        discarding.append(fn.name)
        return sorted(set(discarding))

    @staticmethod
    def _callers() -> list[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(MotorController)))
        callers: list[str] = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name == "move_motor":
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr == "move_motor":
                    callers.append(fn.name)
        return sorted(set(callers))

    def test_the_scan_finds_every_commanding_path(self) -> None:
        """Non-vacuity: the rule below is graded against real callers."""
        callers = self._callers()
        assert {"move_multiple_motors", "_smooth_move", "incremental_move"} <= set(callers), callers

    def test_no_caller_discards_the_outcome(self) -> None:
        assert self._callers_that_discard() == []


class TestWhatIsUnchanged:
    """The reported outcome changed; the packets and the healthy path did not."""

    def test_a_healthy_move_still_reports_success(self, bus: dict[str, Any]) -> None:
        assert _interpolate(bus) is True

    def test_a_healthy_move_writes_one_increment_per_step_per_joint(self, bus: dict[str, Any]) -> None:
        """``range(steps + 1)`` increments for each requested joint, as before."""
        _interpolate(bus)

        commanded = _goal_position_ids(bus["made"][0].writes)
        assert len(commanded) == (_STEPS + 1) * len(_TARGET)
        for motor_id in _IDS.values():
            assert commanded.count(motor_id) == _STEPS + 1

    def test_the_readable_joints_are_still_driven_past_a_silent_one(self, bus: dict[str, Any]) -> None:
        """Reporting the failure must not abandon the joints that can move.

        ``disable_torque`` attempts every motor past a failure for the same
        reason; the return value is what changed, not the trajectory.
        """
        bus["answering"].discard(_IDS["elbow_flex"])

        _interpolate(bus)

        commanded = _goal_position_ids(bus["made"][0].writes)
        assert commanded.count(_IDS["shoulder_pan"]) == _STEPS + 1

    def test_the_readable_joints_are_still_driven_past_a_refused_write(self, bus: dict[str, Any]) -> None:
        bus["refusing"].add(_IDS["elbow_flex"])

        _interpolate(bus)

        commanded = _goal_position_ids(bus["made"][0].writes)
        assert commanded.count(_IDS["shoulder_pan"]) == _STEPS + 1

    def test_an_empty_loop_is_not_reported_as_a_failure(self, bus: dict[str, Any]) -> None:
        """A step count that runs no increments drops nothing, so it is not a failure.

        ``steps <= 0`` is refused by ``_smooth_move_option_error`` before any
        caller of ``pose_tool`` reaches here. Asking for no increments is not the
        same as a joint that could not move, and the fix deliberately does not
        conflate them.
        """
        controller = _connected(bus)

        assert controller._smooth_move(dict(_TARGET), steps=-5, step_delay=_DELAY) is True
        assert _goal_position_ids(bus["made"][0].writes) == []

    def test_the_one_shot_branch_is_untouched(self, bus: dict[str, Any]) -> None:
        """Its outcome reading predates this change; it must still hold."""
        assert _one_shot(bus) is True

        bus["refusing"].update(_IDS.values())
        assert _one_shot(bus) is False
