# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``pose_tool``'s ``steps`` / ``step_delay`` options.

Two properties, and the first is what the second exists to make safe:

1. **The options reach the loop that reads them.** They are documented
   parameters of the tool, but ``move_multiple_motors`` did not accept them and
   invoked ``_smooth_move(positions)`` with neither, so every interpolated move
   ran the hardcoded 20 increments at 0.05s and reported success. The
   observable pinned here is the *requested* pacing - the number of increments
   written and the delay asked of ``time.sleep`` - rather than elapsed wall
   clock, so the assertion is exact on any host.

2. **A value the loop cannot honor is refused, before the port is opened.**
   Once forwarded, both values are consumed on a live servo bus: ``steps``
   divides each motor's travel and bounds the write loop, and ``step_delay`` is
   the pause between goal positions. ``TestWhyTheValuesAreRefused`` measures
   each failure directly against ``_smooth_move`` so the domain is justified by
   what the loop does with the value, not by assertion.

Every test that reaches the motor path takes ``fake_serial`` (or the reading
variant below) and passes an explicit fake ``port``: ``pose_tool``'s ``port``
defaults to ``/dev/ttyACM0``, so a test that omits it drives whatever arm is
plugged into the machine running the suite.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import serial

import strands_robots.tools.pose_tool as pose_mod
from strands_robots.tools.pose_tool import (
    MotorController,
    PoseManager,
    _smooth_move_option_error,
    pose_tool,
)
from strands_robots.utils import positive_count_error, positive_finite_number_error

from .conftest import FakeSerial

# Actions that build an interpolated trajectory, and how each one gets there.
# ``reset_to_home`` passes ``smooth=True`` itself, so it interpolates whatever
# the caller's flag says.
_INTERPOLATING = ("load_pose", "move_multiple", "reset_to_home")

# Counts that cannot bound the write loop or divide the travel.
_UNUSABLE_STEPS: tuple[Any, ...] = (0, -5, 2.7, "4", None, True, False, [4], math.nan)

# Delays that cannot pace the loop. ``0`` is among them: the pause is the
# smoothing, so a zero delay writes every increment as fast as the bus accepts
# it, which is the one-shot move ``smooth=False`` already spells.
_UNUSABLE_DELAYS: tuple[Any, ...] = (0, 0.0, -0.01, math.nan, math.inf, "0.05", None, True, [0.05])

_MOTORS = {"shoulder_pan": 5.0, "elbow_flex": -5.0}


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool through one funnel.

    Several tests deliberately supply values outside the declared types (that is
    the contract under test), and a ``**dict[str, Any]`` splat is not narrowed,
    so routing every call through here states the intent once instead of
    scattering per-call suppressions.

    Args:
        **kwargs: Forwarded verbatim to :func:`pose_tool`.

    Returns:
        The tool result dict.
    """
    return pose_tool(**kwargs)


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _position_packet(raw: int = 0x0800, motor_id: int = 0x01) -> bytes:
    """A Feetech status packet reporting ``raw`` counts for ``motor_id``.

    ``FF FF ID LEN ERR <lo> <hi> CHK``, with the checksum a servo would send so
    the frame passes the verification ``read_motor_position`` performs. Framing
    itself is graded in ``test_feetech_status_packet_framing``.
    """
    body = [motor_id, 0x04, 0x00, raw & 0xFF, (raw >> 8) & 0xFF]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


class _ReadingSerial(FakeSerial):
    """A ``FakeSerial`` that always answers a read with a decodable position.

    ``_smooth_move`` reads the current pose before interpolating and only steps
    the motors it could read, so a source that never answers would make the
    interpolation vacuous.
    """

    def read(self, n: int = 1) -> bytes:
        # Answer as the motor the outgoing packet addressed; a servo bus does,
        # and a fake that always answered as motor 1 would let a read attribute
        # one motor's position to another with no test able to see it.
        asked = self.writes[-1][2] if self.writes else 0x01
        return _position_packet(motor_id=asked)


@pytest.fixture
def reading_serial(monkeypatch: pytest.MonkeyPatch) -> list[_ReadingSerial]:
    """Patch ``serial.Serial`` with an always-answering position source."""
    instances: list[_ReadingSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _ReadingSerial:
        fs = _ReadingSerial(port, baudrate, timeout)
        instances.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


@pytest.fixture
def pacing(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Record the delay each ``time.sleep`` is asked for and every motor write.

    The recorder replaces the sleep rather than shortening it, so a test never
    pays the delay it is asserting about, and the assertion is on the value the
    loop *requested* - which no amount of host load can change.
    """
    seen: dict[str, list[Any]] = {"sleeps": [], "moves": []}
    real_move = MotorController.move_motor

    def _sleep(seconds: Any) -> None:
        seen["sleeps"].append(seconds)

    def _move(self: MotorController, motor_name: str, position_degrees: float) -> bool:
        seen["moves"].append((motor_name, position_degrees))
        return bool(real_move(self, motor_name, position_degrees))

    monkeypatch.setattr(pose_mod.time, "sleep", _sleep)
    monkeypatch.setattr(MotorController, "move_motor", _move)
    return seen


def _stored_pose(cwd_tmp: Any) -> None:
    """Persist a pose named ``target`` for the ``load_pose`` action to load."""
    PoseManager("hw_arm").store_pose("target", dict(_MOTORS))


def _smooth(controller: MotorController, **kwargs: Any) -> bool:
    """Invoke the interpolation loop directly, through one funnel.

    Same reason as :func:`_call`: some of these values are deliberately outside
    the declared types, and a ``**dict[str, Any]`` splat is not narrowed, so the
    intent is stated here once rather than per call.

    Args:
        controller: A connected controller.
        **kwargs: Forwarded to ``_smooth_move`` alongside the target pose.

    Returns:
        Whatever the interpolation loop returns.
    """
    return bool(controller._smooth_move(dict(_MOTORS), **kwargs))


def _drive(action: str, **extra: Any) -> dict[str, Any]:
    """Invoke one interpolating action with the arguments it requires."""
    kwargs: dict[str, Any] = {"action": action, "robot_id": "hw_arm", "port": "/dev/ttyTEST"}
    if action == "load_pose":
        kwargs["pose_name"] = "target"
    if action == "move_multiple":
        kwargs["positions"] = dict(_MOTORS)
    kwargs.update(extra)
    return _call(**kwargs)


class TestTheRequestedPacingReachesTheLoop:
    """The tool's documented options decide the trajectory, not a hardcoded pair."""

    @pytest.mark.parametrize("action", _INTERPOLATING)
    def test_the_requested_step_count_and_delay_are_the_ones_used(
        self, action: str, cwd_tmp: Any, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        _stored_pose(cwd_tmp)
        result = _drive(action, steps=4, step_delay=0.02)
        assert result["status"] == "success", _texts(result)
        # One pause per increment: range(steps + 1).
        assert pacing["sleeps"].count(0.02) == 5, pacing["sleeps"]
        assert 0.05 not in pacing["sleeps"], "the hardcoded default delay was used instead"

    @pytest.mark.parametrize("action", _INTERPOLATING)
    def test_a_finer_request_writes_more_increments_than_a_coarser_one(
        self, action: str, cwd_tmp: Any, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        _stored_pose(cwd_tmp)
        assert _drive(action, steps=3, step_delay=0.01)["status"] == "success"
        coarse = len(pacing["moves"])
        pacing["moves"].clear()
        assert _drive(action, steps=12, step_delay=0.01)["status"] == "success"
        fine = len(pacing["moves"])
        assert fine > coarse, f"{fine} writes for 12 steps vs {coarse} for 3 - the count was dropped"
        # 13 increments vs 4, over the same motors.
        assert fine == coarse * 13 // 4

    def test_omitting_the_options_keeps_the_documented_default_trajectory(
        self, cwd_tmp: Any, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """The default path is unchanged: 20 increments at 0.05s, ~1s of travel."""
        assert _drive("move_multiple")["status"] == "success"
        assert pacing["sleeps"].count(0.05) == 21, pacing["sleeps"]


class TestAnUnusableOptionIsRefusedBeforeThePortOpens:
    """A value the loop cannot honor is reported, not carried onto the bus."""

    @pytest.mark.parametrize("action", _INTERPOLATING)
    @pytest.mark.parametrize("steps", _UNUSABLE_STEPS)
    def test_an_unusable_step_count_is_refused(
        self, action: str, steps: Any, cwd_tmp: Any, fake_serial: list[FakeSerial]
    ) -> None:
        _stored_pose(cwd_tmp)
        result = _drive(action, steps=steps)
        text = _texts(result)
        assert result["status"] == "error", text
        assert "steps" in text and action in text, text
        assert text.isascii(), text
        assert fake_serial == [], "the refused call opened the serial port"

    @pytest.mark.parametrize("action", _INTERPOLATING)
    @pytest.mark.parametrize("step_delay", _UNUSABLE_DELAYS)
    def test_an_unusable_delay_is_refused(
        self, action: str, step_delay: Any, cwd_tmp: Any, fake_serial: list[FakeSerial]
    ) -> None:
        _stored_pose(cwd_tmp)
        result = _drive(action, step_delay=step_delay)
        text = _texts(result)
        assert result["status"] == "error", text
        assert "step_delay" in text and action in text, text
        assert text.isascii(), text
        assert fake_serial == [], "the refused call opened the serial port"

    def test_the_refusal_precedes_reading_the_pose_file(self, cwd_tmp: Any, fake_serial: list[FakeSerial]) -> None:
        """The option is checked before the action's own arguments are resolved."""
        result = _drive("load_pose", pose_name="no_such_pose", steps=0)
        text = _texts(result)
        assert result["status"] == "error"
        assert "steps" in text, text
        assert "not found" not in text, text

    def test_a_usable_pair_is_accepted(
        self, cwd_tmp: Any, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        assert _drive("move_multiple", steps=1, step_delay=1e-6)["status"] == "success"


class TestOnlyTheInterpolatingActionsReadTheOptions:
    """A caller is never refused for a value the requested action ignores."""

    @pytest.mark.parametrize(
        "action,extra",
        [
            ("list_poses", {}),
            ("connect", {}),
            ("read_all", {}),
            ("move_motor", {"motor_name": "elbow_flex", "position": 5.0}),
            ("incremental_move", {"motor_name": "elbow_flex", "delta": 1.0}),
            ("emergency_stop", {}),
        ],
    )
    def test_a_one_shot_action_ignores_the_interpolation_options(
        self, action: str, extra: dict[str, Any], cwd_tmp: Any, reading_serial: list[_ReadingSerial]
    ) -> None:
        result = _drive(action, steps=0, step_delay=-1, **extra)
        text = _texts(result)
        assert "steps" not in text and "step_delay" not in text, text

    def test_move_multiple_with_smooth_false_ignores_them(
        self, cwd_tmp: Any, reading_serial: list[_ReadingSerial]
    ) -> None:
        result = _drive("move_multiple", smooth=False, steps=0, step_delay=-1)
        assert result["status"] == "success", _texts(result)

    def test_reset_to_home_reads_them_even_with_smooth_false(self, cwd_tmp: Any, fake_serial: list[FakeSerial]) -> None:
        """It passes ``smooth=True`` itself, so the caller's flag cannot opt out."""
        result = _drive("reset_to_home", smooth=False, steps=0)
        assert result["status"] == "error", _texts(result)
        assert "steps" in _texts(result)


class TestWhyTheValuesAreRefused:
    """Each refused value measured against the loop that would have received it."""

    @staticmethod
    def _connected(reading_serial: list[_ReadingSerial]) -> MotorController:
        controller = MotorController("/dev/ttyTEST")
        connected, error = controller.connect()
        assert connected, error
        return controller

    def test_zero_steps_divides_by_zero(self, reading_serial: list[_ReadingSerial]) -> None:
        controller = self._connected(reading_serial)
        with pytest.raises(ZeroDivisionError):
            _smooth(controller, steps=0)

    def test_a_negative_step_count_writes_nothing_yet_reports_success(
        self, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """``range(steps + 1)`` is empty, so the move is reported but never made."""
        controller = self._connected(reading_serial)
        assert _smooth(controller, steps=-5) is True
        assert pacing["moves"] == []

    def test_a_boolean_step_count_is_a_full_travel_jump(
        self, reading_serial: list[_ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """``True`` is a single increment - the jump interpolating exists to avoid."""
        controller = self._connected(reading_serial)
        assert _smooth(controller, steps=True) is True
        assert len(pacing["moves"]) == 2 * len(_MOTORS)

    def test_a_fractional_step_count_cannot_bound_the_loop(self, reading_serial: list[_ReadingSerial]) -> None:
        controller = self._connected(reading_serial)
        with pytest.raises(TypeError):
            _smooth(controller, steps=2.7)

    @pytest.mark.parametrize("delay,expected", [(-0.01, ValueError), (math.nan, ValueError), (math.inf, OverflowError)])
    def test_an_unusable_delay_raises_from_sleep(
        self, delay: float, expected: type[BaseException], reading_serial: list[_ReadingSerial]
    ) -> None:
        controller = self._connected(reading_serial)
        with pytest.raises(expected):
            _smooth(controller, steps=1, step_delay=delay)


class TestTheDomainsAreTheSharedOnes:
    """Both options are held to a library-wide domain, not a local rule."""

    @pytest.mark.parametrize("value", (*_UNUSABLE_STEPS, 1, 20, 500))
    def test_steps_is_the_shared_positive_count_domain(self, value: Any) -> None:
        assert _smooth_move_option_error(
            "move_multiple", smooth=True, steps=value, step_delay=0.05
        ) == positive_count_error(value, "steps", "move_multiple")

    @pytest.mark.parametrize("value", (*_UNUSABLE_DELAYS, 1e-6, 0.05, 2.5))
    def test_step_delay_is_the_shared_positive_finite_domain(self, value: Any) -> None:
        assert _smooth_move_option_error(
            "move_multiple", smooth=True, steps=20, step_delay=value
        ) == positive_finite_number_error(value, "step_delay", "move_multiple")

    def test_the_step_count_is_reported_before_the_delay(self) -> None:
        """Both unusable: the caller is told about the count first, deterministically."""
        message = _smooth_move_option_error("move_multiple", smooth=True, steps=0, step_delay=-1)
        assert message is not None and "steps" in message and "step_delay" not in message
