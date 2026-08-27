"""A ``drive`` whose trailing stop was refused reports it, rather than the hold.

A timed or repeated ``drive`` owns its own stop: the shared mobile base and the
two bridges that own their own ``drive`` all follow a non-zero command with a
single zero one, from ``finally``, so the zero goes out even when the main
publish raised. Sending it is not the same as landing it. That zero is a second
command over the same transport, and every shipped graph tool reports a refusal
- a declined operator approval, a rate limit, a transport failure - as an error
envelope rather than by raising, so the refusal was a value that had to be read.
Three of the four drive-owning classes dropped it on the floor inside the
``finally`` and returned the hold's own success.

Measured against the real ``use_rosbridge`` gate with an operator who approves
the hold and declines the stop, on the blocklisted ``/turtle1/cmd_vel``::

    drive(linear=1.0, duration=1.0) -> status="success",
                                       "published 2 message(s) to /turtle1/cmd_vel"
    messages on the wire            -> 2, the last one linear.x = 1.0
    trailing zero                   -> never published

The robot is left moving at the commanded velocity and the caller is told the
drive succeeded - and the agent-facing tool description promises a timed command
"stops automatically afterwards", so a caller reading ``success`` never issues
``stop``. ``RtpsRobot`` reproduces it through the shared base for the same
reason.

``AckermannRosRobot`` already read the verdict and is the reason this is a
consolidation rather than a new rule: it kept ``halt``, compared it against the
hold's own result and returned an error naming the still-live topic. Its own
docstring claims the "same contract as ``RosBridgedRobot.drive``", which was the
reverse of the truth. The rule now lives once, beside the drive contract it
belongs to, and the car's message is unchanged - which is what its three
existing tests in ``tests/mesh/test_ackermann_robot.py`` measure.

Hermetic: the roslibpy WebSocket client and the RTPS writer bundle are doubled
beneath the real tools, so the operator gate and the transport wiring both run
unmodified while nothing reaches a real server.
"""

from __future__ import annotations

import inspect
import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh._mobile_base import LATCHED_VELOCITY, failed_halt_error
from tests.mesh.test_drive_contract_fleet_scope import _drive_owning_mesh_classes
from tests.mesh.test_rosbridge_robot_command_gate import (
    _BLOCKED_CMD_VEL,
    _ZERO_TWIST,
    _published,
    _texts,
    _tool,
    _turtle,
)
from tests.mesh.test_rosbridge_robot_command_gate import (
    _install_doubles as _install_rosbridge_doubles,
)
from tests.mesh.test_rtps_robot_command_gate import (
    _install_doubles as _install_rtps_doubles,
)
from tests.mesh.test_rtps_robot_command_gate import (
    _turtle as _rtps_turtle,
)

#: An operator who approves the hold and then declines the stop that ends it.
#: The gate prompts once per publish - pinned by
#: ``test_the_trailing_zero_of_a_timed_drive_reaches_the_same_operator`` in the
#: rosbridge gate suite - so this is an ordinary sequence of two decisions and
#: not a contrived one.
_APPROVE_THEN_DECLINE = ["y", "n"]

_SUCCESS = {"status": "success", "content": [{"text": "published 20 message(s)"}]}
_FAILURE = {"status": "error", "content": [{"text": "use_ros: publish failed"}]}


def _drive_definers() -> dict[str, str]:
    """Each drive-owning class, mapped to the module file that defines its ``drive``.

    ``_drive_owning_mesh_classes`` is annotated ``dict[str, type]`` and ``drive``
    is not an attribute of a bare ``type``, so the widening happens here once
    instead of at every read.
    """
    owners: dict[str, Any] = dict(_drive_owning_mesh_classes())
    return {name: pathlib.Path(inspect.getsourcefile(cls.drive) or "").name for name, cls in owners.items()}


def _drive_source(module_name: str) -> str:
    """The ``drive`` body defined by ``module_name``."""
    owners: dict[str, Any] = dict(_drive_owning_mesh_classes())
    for cls in owners.values():
        if pathlib.Path(inspect.getsourcefile(cls.drive) or "").name == module_name:
            return inspect.getsource(cls.drive)
    raise AssertionError(f"no drive owner is defined in {module_name}")


def _decides(*answers: str) -> MagicMock:
    context = MagicMock()
    context.interrupt.side_effect = list(answers)
    return context


class TestAnApprovedHoldOverARefusedStopIsReported:
    """The regression: the drive that could not stop itself says so."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_rosbridge_doubles(monkeypatch)

    def test_the_verdict_is_the_refused_stop_not_the_published_hold(self) -> None:
        context = _decides(*_APPROVE_THEN_DECLINE)
        robot = _turtle(publish_rate=2.0)
        result = _tool(robot, "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert context.interrupt.call_count == 2, "the stop never reached the operator"
        assert result["status"] == "error"

    def test_the_refusal_names_the_topic_that_is_still_live(self) -> None:
        """Named where only the refusal can name it.

        The hold's own success text is ``"published N message(s) to <topic>"``,
        so a bare ``topic in text`` passes on the discarding code too - it reads
        the success it is meant to refute. The topic is pinned in the clause the
        refusal builds around it instead.
        """
        context = _decides(*_APPROVE_THEN_DECLINE)
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert f"{_BLOCKED_CMD_VEL}, but the trailing stop" in _texts(result)

    def test_the_refusal_names_the_action_the_caller_has_to_take_next(self) -> None:
        """The imperative, not just the word.

        ``"stop"`` also appears in the refusal's own subject ("the trailing stop
        failed"), so a bare ``"stop" in text`` still passes with the instruction
        removed. The clause that tells the caller what to do is what is pinned.
        """
        context = _decides(*_APPROVE_THEN_DECLINE)
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert "Halt it with stop" in _texts(result)

    def test_the_stops_own_cause_survives_into_the_report(self) -> None:
        """The operator's decision is the cause, and it is what a reader needs."""
        context = _decides(*_APPROVE_THEN_DECLINE)
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert "declined by the operator" in _texts(result)


class TestTheRobotReallyIsStillMoving:
    """Premise: holds either way, and is why the error is the honest verdict.

    A refused stop is not a bookkeeping detail - the velocity the operator
    approved is still the last thing on the wire. Recorded separately because it
    is true of the discarding code as well: what changed is whether the caller
    is told.
    """

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_rosbridge_doubles(monkeypatch)

    def test_the_commanded_velocity_is_the_last_thing_on_the_wire(self) -> None:
        context = _decides(*_APPROVE_THEN_DECLINE)
        _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        published = _published()
        assert published, "the hold was never published, so there is nothing latched"
        assert published[-1][1] != _ZERO_TWIST, "a zero reached the wire after all"
        assert published[-1][1]["linear"]["x"] == 1.0


class TestTheSharedBaseReportsItToo:
    """``RtpsRobot`` reaches the same code through ``MobileBaseRobot.drive``."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_rtps_doubles(monkeypatch)

    def test_a_repeated_command_over_a_refused_stop_is_reported(self) -> None:
        context = _decides(*_APPROVE_THEN_DECLINE)
        result = _rtps_turtle().drive(linear=1.0, count=2, tool_context=context)
        assert context.interrupt.call_count == 2
        assert result["status"] == "error"
        text = " ".join(block.get("text", "") for block in result["content"])
        assert "/turtle1/cmd_vel" in text and "stop" in text


class TestTheRuleItself:
    """The shared verdict-reader, driven directly over every input shape."""

    def test_a_refused_stop_under_a_published_hold_is_reported(self) -> None:
        text = failed_halt_error(_SUCCESS, _FAILURE, topic="/cmd_vel", subject=LATCHED_VELOCITY)
        assert text is not None
        assert "/cmd_vel, but the trailing stop" in text
        assert "Halt it with stop" in text
        assert "use_ros: publish failed" in text

    def test_a_stop_that_landed_reports_nothing(self) -> None:
        assert failed_halt_error(_SUCCESS, _SUCCESS, topic="/cmd_vel", subject=LATCHED_VELOCITY) is None

    def test_no_stop_was_owed_so_there_is_no_verdict_to_read(self) -> None:
        """``None`` is a single-shot command, which latches by contract."""
        assert failed_halt_error(_SUCCESS, None, topic="/cmd_vel", subject=LATCHED_VELOCITY) is None

    def test_a_hold_that_failed_is_the_cause_not_the_stop_it_never_undid(self) -> None:
        assert failed_halt_error(_FAILURE, _FAILURE, topic="/cmd_vel", subject=LATCHED_VELOCITY) is None

    def test_a_stop_that_reported_no_detail_still_reports_the_latch(self) -> None:
        """The latch is the finding; the missing cause must not swallow it."""
        text = failed_halt_error(
            _SUCCESS, {"status": "error", "content": []}, topic="/cmd_vel", subject=LATCHED_VELOCITY
        )
        assert text is not None and "no detail reported" in text

    def test_the_subject_is_the_platforms_own_vocabulary(self) -> None:
        """A car holds a throttle where a differential-drive base holds a velocity."""
        car = failed_halt_error(_SUCCESS, _FAILURE, topic="/servo", subject="the car may still be holding")
        assert car is not None and "the car may still be holding" in car
        assert LATCHED_VELOCITY not in car


class TestWhatIsUnchanged:
    """Controls: every outcome the pre-fix code already got right."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_rosbridge_doubles(monkeypatch)

    def test_a_hold_and_a_stop_the_operator_both_approve_still_succeeds(self) -> None:
        context = _decides("y", "y")
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert result["status"] == "success"
        assert _published()[-1] == (_BLOCKED_CMD_VEL, _ZERO_TWIST)

    def test_a_declined_hold_still_reports_the_hold(self) -> None:
        """Refusing the hold refuses the motion, so its undo has nothing to undo."""
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(
            linear=1.0, duration=1.0, tool_context=_decides("n", "n")
        )
        assert result["status"] == "error"
        assert "declined by the operator" in _texts(result)
        assert _published() == []

    def test_a_single_shot_command_is_not_reinterpreted(self) -> None:
        """No stop is owed, so no verdict exists that could turn success into error."""
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=_decides("y"))
        assert result["status"] == "success"
        assert _published() == [(_BLOCKED_CMD_VEL, {"linear": {"x": 1.0}, "angular": {"z": 0.0}})]

    def test_the_stop_still_goes_out_when_the_hold_itself_failed(self) -> None:
        """The ``finally`` guarantee is untouched: a failed hold still gets its zero."""
        context = _decides("n", "y")
        _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert _published() == [(_BLOCKED_CMD_VEL, _ZERO_TWIST)]


class TestOneOwnerForTheRule:
    """Every class that defines ``drive`` consults the one verdict-reader.

    Derived rather than listed: the defect was three independent implementations
    of one safety rule, two of which had dropped it, so a fifth bridge that
    defines its own ``drive`` has to answer this on arrival instead of shipping
    with the hold's success in place of the stop's refusal.
    """

    def test_every_module_that_defines_drive_reads_the_stops_verdict(self) -> None:
        definers = _drive_definers()
        assert definers, "the drive-owner inventory came back empty"
        for module_name in sorted(set(definers.values())):
            assert "failed_halt_error(" in _drive_source(module_name), (
                f"{module_name} defines drive and does not consult failed_halt_error, "
                "so a refused trailing stop is reported as the hold's success"
            )

    def test_the_rule_is_written_once(self) -> None:
        """One owner, so the platforms cannot drift apart again."""
        package = pathlib.Path(inspect.getfile(failed_halt_error)).parent
        definitions = [
            path.name for path in sorted(package.glob("*.py")) if "def failed_halt_error(" in path.read_text()
        ]
        assert definitions == ["_mobile_base.py"], definitions

    def test_the_inventory_covers_more_than_the_shared_base(self) -> None:
        """Non-vacuity: the survey reaches the classes that own their own drive."""
        definers = set(_drive_definers().values())
        assert {"_mobile_base.py", "rosbridge_robot.py", "ackermann_robot.py"} <= definers, definers


class TestTheProseMatchesTheMeasurement:
    """The guarantee is stated where a reader learns the contract."""

    def test_the_shared_safety_contract_says_the_verdict_is_read(self) -> None:
        import strands_robots.mesh._mobile_base as base

        doc = " ".join((base.__doc__ or "").split())
        assert "verdict is therefore read, not dropped" in doc

    def test_the_helper_states_why_a_refusal_is_a_value_and_not_a_raise(self) -> None:
        doc = " ".join((failed_halt_error.__doc__ or "").split())
        assert "error envelope rather than by raising" in doc
