# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pose_tool`` must refuse a ``smooth`` posture it can only misread.

``smooth`` selects between two trajectories towards the same joint targets, not
a magnitude: interpolate over ``steps * step_delay`` seconds, or write each goal
position once. It was read by truthiness, and the two undeclared halves invert
in opposite directions - measured on ``b882372``, ``move_multiple`` with two
motors and the default ``steps=20``:

* ``smooth=0`` (also ``""``, ``None``, ``[]``) wrote **2** goal positions with no
  pause, against **42** over 21 increments for ``smooth=True``. The flag defaults
  to ``True``, so a falsy non-boolean silently *removes* the interpolation the
  caller never asked to leave, and what reaches the servo is a single write to
  the far end of the travel - the full-travel jump this module already refuses
  ``steps=True`` for, on the grounds that it is "exactly the path a caller asking
  to interpolate wanted to avoid".
* ``smooth="false"`` (also ``"no"``, ``"off"``, ``"0"``) is truthy, so it selected
  the interpolating branch the word asks to skip.
* Because the flag also decides whether ``steps`` / ``step_delay`` are read at
  all, ``smooth="false", steps=0`` was refused with ``move_multiple: steps must
  be a positive integer, got 0.`` - sending the caller to correct an option their
  posture said nobody would read.

All four rows reported ``status="success"`` (or, in the last, an error about the
wrong parameter), so the wrong trajectory was indistinguishable from the right
one.

The flag is now held to the shared
:func:`~strands_robots.utils.boolean_flag_error` domain, ahead of the
``steps`` / ``step_delay`` check so a bad flag is named as the flag, and only for
the two actions that read it - the scoping rule those numeric options already
follow (``tests/tools/test_pose_tool_interpolation_options.py``).

Every test that reaches the motor path takes a serial double and passes an
explicit fake ``port``: ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``, so
a test that omits it drives whatever arm is plugged into the machine running the
suite.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.pose_tool as pose_mod
from strands_robots.tools.pose_tool import (
    MotorController,
    PoseManager,
    _smooth_posture_error,
    pose_tool,
)
from strands_robots.utils import boolean_flag_error

from .conftest import FakeSerial, ReadingSerial


@pytest.fixture(autouse=True)
def _pre_approve_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests grade what a motion does once admitted; the operator gate (F-010) is graded in
    ``test_pose_tool_gates_bus_writes.py``, so it is pre-approved here."""
    monkeypatch.setenv("STRANDS_POSE_COMMAND_ALLOW", "*")


# The two actions that consult the caller's flag. ``reset_to_home`` interpolates
# unconditionally and passes its own ``smooth=True``; every other action moves in
# one shot. Neither reads this flag, so neither may be refused for it.
_READS_THE_FLAG = ("load_pose", "move_multiple")
_IGNORES_THE_FLAG = ("reset_to_home", "read_all", "connect", "list_poses")

# Truthy: these select the interpolating branch the word asks to skip.
_TRUTHY_NON_BOOLEANS: tuple[Any, ...] = ("false", "no", "off", "0", 1, 2, math.nan)
# Falsy: these drop the interpolation, and none is a declared spelling of that.
_FALSY_NON_BOOLEANS: tuple[Any, ...] = (0, 0.0, "", None, [], {})

_MOTORS = {"shoulder_pan": 5.0, "elbow_flex": -5.0}


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool through one funnel.

    The values under test are deliberately outside the declared ``bool``, which
    is the point; mypy does not narrow a splatted ``dict[str, Any]``, so routing
    every call through here states that once instead of suppressing it per call.
    """
    return pose_tool(**kwargs)


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


@pytest.fixture
def pacing(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Record every motor write and the delay each ``time.sleep`` is asked for.

    The recorder replaces the sleep rather than shortening it, so the assertion
    is on the pacing the loop *requested*, which no host load can change.
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


def _drive(action: str, **extra: Any) -> dict[str, Any]:
    """Invoke one action with the arguments it requires and a fake port."""
    kwargs: dict[str, Any] = {"action": action, "robot_id": "hw_arm", "port": "/dev/ttyTEST"}
    if action == "load_pose":
        kwargs["pose_name"] = "target"
    if action == "move_multiple":
        kwargs["positions"] = dict(_MOTORS)
    kwargs.update(extra)
    return _call(**kwargs)


@pytest.fixture
def stored_pose(cwd_tmp: Any) -> None:
    """Persist a pose named ``target`` for the ``load_pose`` action to load."""
    PoseManager("hw_arm").store_pose("target", dict(_MOTORS))


class TestAPostureThatIsNotABooleanIsRefused:
    """Neither half is read as a trajectory, and the port stays closed."""

    @pytest.mark.parametrize("action", _READS_THE_FLAG)
    @pytest.mark.parametrize("smooth", (*_TRUTHY_NON_BOOLEANS, *_FALSY_NON_BOOLEANS))
    def test_an_undeclared_spelling_is_refused_naming_the_flag(
        self, action: str, smooth: Any, stored_pose: None, fake_serial: list[FakeSerial]
    ) -> None:
        result = _drive(action, smooth=smooth)
        text = _texts(result)
        assert result["status"] == "error", text
        assert "smooth" in text and action in text, text
        assert text.isascii(), text
        assert fake_serial == [], "the refused call opened the serial port"

    def test_the_refusal_precedes_reading_the_pose_file(self, cwd_tmp: Any, fake_serial: list[FakeSerial]) -> None:
        """The flag is checked before the action's own arguments are resolved."""
        result = _drive("load_pose", pose_name="no_such_pose", smooth="false")
        text = _texts(result)
        assert result["status"] == "error"
        assert "smooth" in text and "not found" not in text, text

    def test_the_flag_is_named_before_the_options_it_gates(self) -> None:
        """A bad flag is reported as the flag, not as the option it switches on.

        ``smooth`` decides whether ``steps`` / ``step_delay`` are read at all, so
        checked second it produced a refusal naming an option the caller's
        posture said nobody would read.
        """
        result = _drive("move_multiple", smooth="false", steps=0, step_delay=-1)
        text = _texts(result)
        assert result["status"] == "error"
        assert "smooth" in text, text
        assert "steps" not in text and "step_delay" not in text, text


class TestWhyEachHalfIsRefused:
    """Each half measured against the branch it would have selected.

    ``MotorController.move_multiple_motors`` is the branch point, and it is left
    reading the flag it is handed - the domain is enforced once, at the tool,
    where the caller supplies it, exactly as ``steps`` and ``step_delay`` are.
    Driving it directly is what shows what the refused values did.
    """

    @staticmethod
    def _connected(reading_serial: list[ReadingSerial]) -> MotorController:
        controller = MotorController("/dev/ttyTEST")
        connected, error = controller.connect()
        assert connected, error
        return controller

    @pytest.mark.parametrize("smooth", _FALSY_NON_BOOLEANS)
    def test_a_falsy_spelling_is_a_single_full_travel_write(
        self, smooth: Any, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """One goal position per motor, straight to the target, and no pause."""
        controller = self._connected(reading_serial)
        assert controller.move_multiple_motors(dict(_MOTORS), smooth, steps=20, step_delay=0.05) is True
        assert len(pacing["moves"]) == len(_MOTORS), pacing["moves"]
        assert pacing["sleeps"] == []

    @pytest.mark.parametrize("smooth", _TRUTHY_NON_BOOLEANS)
    def test_a_truthy_opt_out_spelling_interpolates_anyway(
        self, smooth: Any, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """The word asks to go straight to the target; 21 increments are written."""
        controller = self._connected(reading_serial)
        assert controller.move_multiple_motors(dict(_MOTORS), smooth, steps=20, step_delay=0.05) is True
        assert len(pacing["moves"]) == 21 * len(_MOTORS), len(pacing["moves"])
        assert pacing["sleeps"].count(0.05) == 21


class TestTheDeclaredSpellingsStillSelectBothBranches:
    """The fix refuses the undeclared halves and nothing else."""

    def test_true_interpolates(
        self, stored_pose: None, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        assert _drive("move_multiple", smooth=True, steps=4, step_delay=0.02)["status"] == "success"
        assert len(pacing["moves"]) == 5 * len(_MOTORS)

    def test_false_goes_straight_to_the_targets(
        self, stored_pose: None, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        assert _drive("move_multiple", smooth=False)["status"] == "success"
        assert len(pacing["moves"]) == len(_MOTORS)
        assert pacing["sleeps"] == []

    def test_a_numpy_boolean_is_a_boolean(
        self, stored_pose: None, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """The shared domain accepts ``np.bool_``, and so does this surface."""
        assert _drive("move_multiple", smooth=np.False_)["status"] == "success"
        assert len(pacing["moves"]) == len(_MOTORS)

    def test_omitting_the_flag_keeps_the_default_interpolated_move(
        self, stored_pose: None, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        assert _drive("move_multiple")["status"] == "success"
        assert pacing["sleeps"].count(0.05) == 21


class TestOnlyTheActionsThatReadTheFlagAreRefused:
    """A caller is never refused for a value the requested action ignores."""

    @pytest.mark.parametrize("action", _IGNORES_THE_FLAG)
    @pytest.mark.parametrize("smooth", ("false", 0))
    def test_an_action_that_ignores_the_flag_is_not_refused_for_it(
        self, action: str, smooth: Any, stored_pose: None, reading_serial: list[ReadingSerial]
    ) -> None:
        assert "smooth" not in _texts(_drive(action, smooth=smooth)), action

    def test_reset_to_home_still_interpolates_whatever_the_flag_says(
        self, stored_pose: None, reading_serial: list[ReadingSerial], pacing: dict[str, list[Any]]
    ) -> None:
        """It supplies its own ``smooth=True``, so the caller's flag is unread."""
        assert _drive("reset_to_home", smooth="false", steps=3, step_delay=0.02)["status"] == "success"
        # One pause per increment: range(steps + 1). The read of the current pose
        # pauses 0.01s per motor, so the asserted delay is deliberately not that.
        assert pacing["sleeps"].count(0.02) == 4, pacing["sleeps"]


class TestTheDomainIsTheSharedOne:
    """The accepted spellings cannot drift from the library-wide flag domain."""

    @pytest.mark.parametrize("value", (*_TRUTHY_NON_BOOLEANS, *_FALSY_NON_BOOLEANS, True, False, np.True_))
    def test_the_message_is_the_shared_flag_domain_verbatim(self, value: Any) -> None:
        assert _smooth_posture_error("move_multiple", value) == boolean_flag_error(value, "smooth", "move_multiple")

    @pytest.mark.parametrize("action", _IGNORES_THE_FLAG)
    def test_an_action_that_ignores_the_flag_yields_no_error(self, action: str) -> None:
        assert _smooth_posture_error(action, "false") is None

    def test_every_boolean_parameter_of_the_tool_is_checked(self) -> None:
        """A flag added to the signature cannot skip the domain unnoticed."""
        flags = {
            name
            for name, param in inspect.signature(pose_tool).parameters.items()
            if param.annotation in (bool, "bool")
        }
        assert flags, "no boolean parameter found - the roster read is broken"
        assert flags == {"smooth"}, flags
        assert all(_smooth_posture_error("move_multiple", None) is not None for _ in flags)
