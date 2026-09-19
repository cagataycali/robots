# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the degree-valued targets ``pose_tool`` drives a joint to.

``position``, ``delta`` and the values of ``positions`` all reach a servo the
same way: ``FeetechBus.to_counts`` scales them onto the 12-bit ``Goal_Position``
register against the travel this arm's calibration measured. That conversion is
on the far side of the operator's approval and of the open port, which is why
this module measures the refusals against it instead of only asserting on them:

* **A target outside the travel has no encoding at all.**
  ``TestWhyTheTargetsAreRefused`` shows ``nan``, ``inf`` and an out-of-range
  target each refused by the conversion rather than folded onto an end stop.
  Reaching that refusal costs a motion the operator has already approved and a
  port already open, and the caller learns only that the joint "failed".

* **Refusing early is what keeps the caller's own value in the message.** The
  success text echoes the *requested* value, so before the guard a move to
  ``nan`` reported ``"Moved shoulder_pan to nan deg"``. That is the property
  ``TestTheBusIsNotTouched`` pins: a refused target produces no write at all.

The two deferrals on the ``delta`` path are pinned here as well, because each is
only sound if the thing it defers TO refuses. An unknown motor has no travel to
bound a displacement against, and ``incremental_move``'s own position read is
what refuses it; a displacement *inside* the travel can still compute an absolute
target outside the range, and the conversion is what refuses that.

The domain itself is delegated to :func:`~strands_robots.utils.finite_number_error`
so an off-type or non-finite target is reported in the words every other surface
uses; only the per-joint bounds are decided in this module, because they are a
property of the arm it drives - measured per arm, and read from the same bus
the target is converted through. That split is asserted in
``TestTheBoundsHaveOneAuthority`` rather than left to convention.

Every test that reaches the motor path takes ``fake_serial`` and passes an
explicit fake ``port``: ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``,
so a test that omits it drives whatever arm is plugged into the machine running
the suite.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest
import serial

from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
from strands_robots.tools.pose_tool import (
    _TARGET_OPTION_BY_ACTION,
    MotorController,
    _joint_delta_error,
    _joint_target_error,
    _pose_target_error,
    _units,
    pose_tool,
)
from strands_robots.utils import finite_number_error

from .conftest import FakeSerial


@pytest.fixture(autouse=True)
def _pre_approve_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests grade what a motion does once admitted; the operator gate (F-010) is graded in
    ``test_pose_tool_gates_bus_writes.py``, so it is pre-approved here."""
    monkeypatch.setenv("STRANDS_POSE_COMMAND_ALLOW", "*")


_PORT = "/dev/fake-pose-target"

# A joint with a symmetric range, and the gripper, which is configured 0-100 and
# is the one motor whose targets are a percentage rather than degrees.
_JOINT = "shoulder_pan"
_JOINT_RANGE = (-180, 180)

# The travel the guards read for an arm with no calibration on disk: the servo's
# whole rotation, and the whole of the gripper's percent domain. Built from the
# bus the tool itself builds, so this module restates no bound.
_TRAVEL = {name: _units(_PORT).value_bounds(name) for name in SO_ARM_MOTORS}

# Targets no joint can be driven to. Each is refused for one of two reasons -
# it is not a finite number, or it is outside the joint's configured travel -
# and ``TestWhyTheTargetsAreRefused`` measures both against the conversion.
_UNUSABLE_TARGETS: tuple[Any, ...] = (
    math.nan,
    math.inf,
    -math.inf,
    5000,
    -5000,
    180.5,
    True,
    False,
    "90",
    [90],  # a list is not a scalar target
    10**400,
)

# Targets inside the joint's travel, which must keep working unchanged.
_USABLE_TARGETS: tuple[Any, ...] = (0, 0.0, 90.0, -90.0, 180, -180, 12.5)


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


def _goal_position(motor_name: str, degrees: Any) -> int:
    """The ``Goal_Position`` ``degrees`` is encoded as.

    Raises whatever the conversion raises for a value it cannot represent, which
    is what several cells below are about.
    """
    return MotorController(_PORT).units.to_counts(motor_name, degrees)


class TestWhyTheTargetsAreRefused:
    """The domain is justified by what the conversion does with the value."""

    @pytest.mark.parametrize("target", [math.nan, math.inf, 5000, 180.5, -math.inf, -5000, -180.5, 10**400])
    def test_a_target_outside_the_travel_has_no_encoding(self, target):
        """The conversion refuses it, so what waits past the guard is a failure.

        Not a wrong count - none exists. Getting there is the whole reason for
        refusing early: the operator has approved the motion and the port is
        open, and the joint is then reported as simply "failed".
        """
        with pytest.raises((ValueError, OverflowError, TypeError)):
            _goal_position(_JOINT, target)

    def test_nan_passes_every_comparison_a_bound_could_make(self):
        """So only an explicit check refuses it; a range test never will."""
        low, high = _TRAVEL[_JOINT]
        assert not low <= math.nan <= high
        assert not math.nan < low
        assert not math.nan > high
        assert _joint_target_error("move_motor", "position", _JOINT, math.nan, _TRAVEL) is not None

    def test_a_bool_would_have_been_read_as_one_degree(self):
        """``True`` is an ``int`` subclass, so it encodes as a real 1-degree move."""
        assert _goal_position(_JOINT, True) == _goal_position(_JOINT, 1)

    @pytest.mark.parametrize("target", _USABLE_TARGETS)
    def test_an_in_range_target_keeps_an_encoding_of_its_own(self, target):
        """An accepted target is distinguishable from the end stops."""
        encoded = _goal_position(_JOINT, target)
        assert 0 <= encoded <= 4095
        if target not in _JOINT_RANGE:
            assert encoded not in (0, 4095)


class TestTheBusIsNotTouched:
    """A refused target produces no serial write, on any of the three actions."""

    @pytest.mark.parametrize("target", _UNUSABLE_TARGETS)
    def test_move_motor_refuses_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=target)
        assert result["status"] == "error"
        assert "position" in _texts(result)
        assert fake_serial == [], "the port was opened for a target that cannot be honored"

    @pytest.mark.parametrize("target", _UNUSABLE_TARGETS)
    def test_move_multiple_refuses_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_multiple", port=_PORT, positions={_JOINT: target})
        assert result["status"] == "error"
        assert f"positions[{_JOINT!r}]" in _texts(result)
        assert fake_serial == []

    @pytest.mark.parametrize("target", [math.nan, math.inf, True, "90", [90], 10**400])
    def test_incremental_move_refuses_a_non_finite_delta_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="incremental_move", port=_PORT, motor_name=_JOINT, delta=target)
        assert result["status"] == "error"
        assert "delta" in _texts(result)
        assert fake_serial == []

    def test_a_delta_larger_than_the_full_travel_is_refused(self, fake_serial, cwd_tmp):
        """No starting position could honor it, so it is unhonorable by construction."""
        span = _JOINT_RANGE[1] - _JOINT_RANGE[0]
        result = _call(action="incremental_move", port=_PORT, motor_name=_JOINT, delta=span + 1)
        assert result["status"] == "error"
        assert f"at most {span} degrees in magnitude" in _texts(result)
        assert fake_serial == []

    def test_the_refusal_names_the_motor_and_its_travel(self, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=5000)
        text = _texts(result)
        assert "[-180, 180] degrees" in text
        assert f"'{_JOINT}'" in text
        assert "5000" in text

    def test_the_gripper_is_quoted_as_a_percentage(self, fake_serial, cwd_tmp):
        """Its configured range is 0-100, so "degrees" would be the wrong word."""
        result = _call(action="move_motor", port=_PORT, motor_name="gripper", position=200)
        assert "[0, 100] percent" in _texts(result)


class TestUsableTargetsStillReachTheServo:
    """The guard refuses; it must not start refusing what already worked."""

    @pytest.mark.parametrize("target", _USABLE_TARGETS)
    def test_move_motor_still_writes_the_goal_position(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=target)
        assert result["status"] == "success"
        assert len(fake_serial) == 1
        written = fake_serial[0].writes
        assert len(written) == 1
        # Feetech INST_WRITE to Goal_Position (0x2A), little-endian payload.
        encoded = _goal_position(_JOINT, target)
        assert written[0][4] == 0x03
        assert written[0][5] == 0x2A
        assert written[0][6] == encoded & 0xFF
        assert written[0][7] == (encoded >> 8) & 0xFF

    def test_move_multiple_still_drives_every_motor(self, fake_serial, cwd_tmp):
        result = _call(
            action="move_multiple",
            port=_PORT,
            positions={_JOINT: 10.0, "elbow_flex": -20.0},
            smooth=False,
        )
        assert result["status"] == "success"
        assert len(fake_serial[0].writes) == 2

    def test_a_target_exactly_on_each_bound_is_accepted(self, fake_serial, cwd_tmp):
        """The bounds are inclusive - they are reachable positions, not limits."""
        for bound in _JOINT_RANGE:
            assert (
                _pose_target_error(
                    "move_motor", motor_name=_JOINT, position=bound, delta=None, positions=None, travel=_TRAVEL
                )
                is None
            )


class TestOnlyTheTargetTheActionReadsIsChecked:
    """A caller is never refused for a value the requested action never looks at."""

    @pytest.mark.parametrize("action", ["read_all", "list_poses", "connect", "read_position"])
    def test_an_action_reading_no_target_is_not_refused(self, action):
        assert (
            _pose_target_error(
                action,
                motor_name=_JOINT,
                position=math.nan,
                delta=math.nan,
                positions={_JOINT: math.nan},
                travel=_TRAVEL,
            )
            is None
        )

    def test_move_motor_ignores_an_unusable_delta(self):
        """It commands an absolute position; ``delta`` is not its parameter."""
        assert (
            _pose_target_error(
                "move_motor", motor_name=_JOINT, position=0.0, delta=math.nan, positions=None, travel=_TRAVEL
            )
            is None
        )

    def test_incremental_move_ignores_an_unusable_position(self):
        assert (
            _pose_target_error(
                "incremental_move", motor_name=_JOINT, position=math.nan, delta=0.0, positions=None, travel=_TRAVEL
            )
            is None
        )

    @pytest.mark.parametrize("action", list(_TARGET_OPTION_BY_ACTION))
    def test_an_absent_target_is_left_to_the_required_check(self, action):
        """The action reports the whole missing pair; this guard must not pre-empt it."""
        assert (
            _pose_target_error(action, motor_name=_JOINT, position=None, delta=None, positions=None, travel=_TRAVEL)
            is None
        )

    def test_a_missing_position_still_reports_the_required_pair(self, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT)
        assert result["status"] == "error"
        assert "required" in _texts(result)


class TestTheBoundsHaveOneAuthority:
    """The servo and the guard must not read two copies of the same travel."""

    @pytest.mark.parametrize("joint", sorted(SO_ARM_MOTORS))
    def test_the_bound_is_the_last_value_the_conversion_can_encode(self, joint):
        """Accepted exactly when encodable - one rule, not two.

        The guard and the servo read the same
        :class:`~strands_robots.drivers.feetech.bus.FeetechBus`, so a target this
        domain accepts has a count and one it refuses has none. A second copy of
        the bounds could accept a target the conversion then refuses on the far
        side of the operator's approval.
        """
        bus = MotorController(_PORT).units
        low, high = _TRAVEL[joint]
        assert bus.value_bounds(joint) == (low, high)

        assert 0 <= bus.to_counts(joint, low) <= bus.motors[joint].resolution
        assert 0 <= bus.to_counts(joint, high) <= bus.motors[joint].resolution
        for outside in (low - 1, high + 1):
            with pytest.raises(ValueError):
                bus.to_counts(joint, outside)

    def test_the_shared_domain_owns_finiteness_and_type(self):
        """Only the per-joint bounds are decided here; the rest is delegated.

        Asserted as an equality with the shared helper's own text so the two
        cannot drift into reporting the same value in different words.

        ``None`` is excluded deliberately: it is the omitted-target spelling,
        which the action's own required check reports as a missing pair. That
        exception is pinned by
        ``TestOnlyTheTargetTheActionReadsIsChecked::test_an_absent_target_is_left_to_the_required_check``.
        """
        for value in (math.nan, math.inf, True, "90", [90], 10**400):
            expected = finite_number_error(value, "position", "move_motor")
            if expected is None:
                continue
            assert (
                _pose_target_error(
                    "move_motor", motor_name=_JOINT, position=value, delta=None, positions=None, travel=_TRAVEL
                )
                == expected
            )


class TestNoTargetSurfaceDrifts:
    """A target parameter added to the tool cannot skip the guard silently."""

    def test_every_declared_target_parameter_is_routed(self):
        """Each ``float | None`` / ``dict[str, float] | None`` option is covered.

        ``steps`` and ``step_delay`` are excluded by annotation rather than by
        name: they are ``int`` and ``float`` with defaults, never ``| None``,
        because they are interpolation options with their own guard.
        """
        routed = set(_TARGET_OPTION_BY_ACTION.values())
        declared = {
            name
            for name, param in inspect.signature(pose_tool.__wrapped__).parameters.items()
            if str(param.annotation) in ("float | None", "dict[str, float] | None")
        }
        assert declared, "the signature probe matched nothing - it has stopped testing anything"
        assert declared == routed, f"unrouted joint-target parameters: {sorted(declared - routed)}"

    def test_every_routed_action_is_a_real_action(self):
        """A typo in the map would silently disable the guard for that action."""
        doc = pose_tool.__wrapped__.__doc__ or ""
        for action in _TARGET_OPTION_BY_ACTION:
            assert f'"{action}"' in doc


class TestNeighbouringTargetProducersStayOutOfScope:
    """The boundary of this change, pinned so it narrows deliberately.

    ``reset_to_home`` is the one remaining producer of a clamped target that
    comes from somewhere other than the caller's arguments, and it is not
    reachable from the guard above: it supplies its own literal targets, and
    every one of them is inside its joint's travel.

    A **stored pose** used to be listed here too, deferred to
    ``PoseManager.validate_pose``. That deferral did not hold: ``validate_pose``
    consults the pose's own optional ``safety_bounds``, which no caller of
    ``store_pose`` supplies, so it answered "No safety bounds defined" for every
    pose this tool writes. ``load_pose`` now routes its stored positions through
    :func:`_joint_target_error` as well, and
    ``test_pose_tool_stored_pose_target_domain`` grades that path.

    The conversion is therefore the last line rather than a clamp. It is
    unreachable from ``move_motor`` / ``move_multiple``, whose targets are
    absolute and are held to the joint's travel - but NOT from
    ``incremental_move``, whose delta is held to the full span instead, so a
    displacement inside that span can still compute an absolute target outside
    the travel. ``TestTheComputedTargetDeferralHolds`` measures that path.
    """

    def test_a_target_past_the_travel_is_refused_by_the_conversion_not_clamped(self):
        """It has no count, so no end-stop command can be fabricated from it."""
        with pytest.raises(ValueError, match="outside the travel the encoder can hold"):
            _goal_position(_JOINT, 5000)

    def test_load_pose_is_not_routed_through_the_target_guard(self):
        assert "load_pose" not in _TARGET_OPTION_BY_ACTION
        assert "reset_to_home" not in _TARGET_OPTION_BY_ACTION

    def test_an_unknown_motor_is_left_to_the_existing_path(self, fake_serial, cwd_tmp):
        """No configured range means no bounds to check it against."""
        assert (
            _pose_target_error(
                "move_motor", motor_name="no_such_joint", position=5000, delta=None, positions=None, travel=_TRAVEL
            )
            is None
        )

    def test_an_unknown_motor_with_a_non_finite_target_is_still_refused(self):
        """Finiteness needs no range, so the shared domain still applies."""
        assert (
            _pose_target_error(
                "move_motor", motor_name="no_such_joint", position=math.nan, delta=None, positions=None, travel=_TRAVEL
            )
            is not None
        )


def _position_packet(raw: int, motor_id: int = 0x01) -> bytes:
    """A Feetech status packet reporting ``raw`` counts for ``motor_id``.

    ``FF FF ID LEN ERR <lo> <hi> CHK``, with the checksum a servo would send so
    the frame passes the verification ``read_motor_position`` performs. Framing
    itself is graded in ``test_feetech_status_packet_framing``.
    """
    body = [motor_id, 0x04, 0x00, raw & 0xFF, (raw >> 8) & 0xFF]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


# A joint parked near the upper end of its travel, and a displacement inside the
# full span but *larger than either endpoint*. That is the value only the travel
# rule accepts, so it distinguishes this domain from one written against the
# endpoints; together they also compute an absolute target outside the range,
# which is the case the delta domain deliberately does not bound.
_NEAR_UPPER_RAW = 3980
_INSIDE_TRAVEL_DELTA = 300


class _ReadingSerial(FakeSerial):
    """A ``FakeSerial`` that always answers a read with a decodable position.

    ``incremental_move`` reads the current position before commanding anything,
    so a source that never answers refuses every motor - configured or not - and
    an assertion that an unknown motor reached no servo would hold for the wrong
    reason.
    """

    def read(self, n: int = 1) -> bytes:
        # Answer as the motor the outgoing packet addressed; a servo bus does,
        # and a fake that always answered as motor 1 would let a read attribute
        # one motor's position to another with no test able to see it.
        asked = self.writes[-1][2] if self.writes else 0x01
        return _position_packet(_NEAR_UPPER_RAW, motor_id=asked)


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


def _goal_positions(instances: list[_ReadingSerial]) -> list[int]:
    """Every ``Goal_Position`` value that reached the bus, decoded from the packets.

    A goal write is ``INST_WRITE`` (``0x03``) whose first parameter is the
    ``Goal_Position`` address (``0x2A``), with the value little-endian after it.
    Reads share the bus, so the payload is what distinguishes a command from a
    query.

    Args:
        instances: The recording serial stand-ins the fixture handed out.

    Returns:
        The commanded goal positions, in the order they were written.
    """
    goals: list[int] = []
    for fake in instances:
        for packet in fake.writes:
            if len(packet) >= 9 and packet[4] == 0x03 and packet[5] == 0x2A:
                goals.append(packet[6] | (packet[7] << 8))
    return goals


class TestTheUnknownMotorDeferralHolds:
    """A displacement for a motor with no configured travel, and what refuses it.

    :func:`_joint_delta_error` returns ``None`` for a motor absent from
    the travel map: there is no travel to bound a displacement
    against, so it has nothing to say and defers - exactly as its sibling
    :func:`_joint_target_error` does for an absolute target.

    A deferral is only sound if the thing it defers TO refuses, and that half
    was unasserted. The branch itself was unexecuted by the whole suite, so a
    change making an unconfigured motor commandable through the delta path would
    have left every test green.
    """

    def test_an_unknown_motor_has_no_travel_to_bound_the_delta_against(self):
        """The domain defers rather than inventing a bound it cannot know."""
        assert _joint_delta_error("incremental_move", "no_such_joint", 5000, _TRAVEL) is None

    def test_both_helpers_defer_for_the_same_absent_configuration(self):
        """Whatever the domain does here it does for the absolute target too."""
        assert _joint_target_error("move_motor", "position", "no_such_joint", 5000, _TRAVEL) is None
        assert _joint_delta_error("incremental_move", "no_such_joint", 5000, _TRAVEL) is None

    def test_finiteness_still_applies_without_a_configured_range(self):
        """Only the per-joint bound needs a configuration; the shared domain does not."""
        assert _joint_delta_error("incremental_move", "no_such_joint", math.nan, _TRAVEL) is not None

    def test_the_action_refuses_the_unknown_motor_without_commanding_it(self, reading_serial, cwd_tmp):
        """The deferral's target: a read that cannot address an unconfigured motor."""
        result = _call(action="incremental_move", motor_name="no_such_joint", delta=5000, port=_PORT)
        assert result["status"] == "error"
        assert "no_such_joint" in _texts(result)
        assert _goal_positions(reading_serial) == []

    def test_a_configured_motor_takes_the_same_call_to_the_servo(self, reading_serial, cwd_tmp):
        """The refusal above is about the motor, not about the reading source."""
        result = _call(action="incremental_move", motor_name=_JOINT, delta=-90, port=_PORT)
        assert result["status"] == "success"
        assert _goal_positions(reading_serial) != []


class TestTheComputedTargetDeferralHolds:
    """A displacement inside the travel can still compute a target outside the range.

    The delta is bounded by the joint's *full travel* rather than by its
    endpoints, because a displacement is relative and the endpoints are not. So
    ``current + delta`` can leave the travel for a delta this domain accepts, and
    the conversion is what refuses it - making ``incremental_move`` the one
    caller-driven path that reaches that refusal. The joint is then reported as
    uncommanded, where a clamp would have driven it to the end stop and echoed
    the displacement back as if it had been honored.
    """

    def test_a_displacement_inside_the_full_travel_is_accepted(self):
        """The premise: this delta is one the domain has no reason to refuse."""
        span = _JOINT_RANGE[1] - _JOINT_RANGE[0]
        assert abs(_INSIDE_TRAVEL_DELTA) < span
        # And larger than either endpoint, so a domain written against those
        # would refuse it: this is what makes the travel rule observable.
        assert abs(_INSIDE_TRAVEL_DELTA) > _JOINT_RANGE[1]
        assert _joint_delta_error("incremental_move", _JOINT, _INSIDE_TRAVEL_DELTA, _TRAVEL) is None

    def test_the_computed_absolute_target_leaves_the_travel(self):
        """The premise for the refusal: the sum is past the end of the travel."""
        start = MotorController(_PORT).units.to_value(_JOINT, _NEAR_UPPER_RAW)
        assert start + _INSIDE_TRAVEL_DELTA > _TRAVEL[_JOINT][1]

    def test_the_joint_is_reported_uncommanded_rather_than_driven_to_the_stop(self, reading_serial, cwd_tmp):
        """No goal position is written, and the caller is told the move failed."""
        result = _call(action="incremental_move", motor_name=_JOINT, delta=_INSIDE_TRAVEL_DELTA, port=_PORT)
        assert result["status"] == "error"
        assert _JOINT in _texts(result)
        assert _goal_positions(reading_serial) == []
