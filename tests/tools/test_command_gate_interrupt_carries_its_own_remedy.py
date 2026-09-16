"""The gate's interrupt tells a headless script how to answer it.

``agent("move the wrist")`` on a real arm returns a PAUSED result: the operator
gate raised an interrupt and the SDK handed it back for someone to answer. A
script that does ``print(agent(...))`` sees the interrupt list's repr - the
question, the action, the target - and nothing that says the run is paused or
how to resume it (the owner's four-line script in the README ends there). The
``reason`` dict is the one thing that script is certainly printing, so it now
carries ``how_to_answer``: the exact ``agent([{"interruptResponse": ...}])``
form, what 'y' means, and the pre-approval line for a script with no operator.
Every tool on the shared gate gets the same line.

The pre-approval half names the variable and the entry outright rather than a
``*_COMMAND_ALLOW`` glob, because a script can derive neither: the three ROS
transports share ``STRANDS_ROS2_COMMAND_ALLOW`` (no ``use_rosbridge`` in the
name), and the entry is the *action* for the arm tools but the *target* for
``use_unitree`` and the ROS transports (``sport.Move``, ``/cmd_vel``). The last
test here is the round trip: whatever the line says to set is set, and the
identical call then proceeds without asking anyone.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from strands.types.tools import ToolContext

from strands_robots.tools import _command_gate as gate_mod
from strands_robots.tools._command_gate import COMMAND_ALLOW_ENV as ROS_ALLOW
from strands_robots.tools._command_gate import allow_entry, gate_command, gate_motion, how_to_answer
from strands_robots.tools.g1.use_unitree import COMMAND_ALLOW_ENV as UNITREE_ALLOW
from strands_robots.tools.g1.use_unitree import _gate as unitree_gate
from strands_robots.tools.pose_tool import COMMAND_ALLOW_ENV as POSE_ALLOW
from strands_robots.tools.pose_tool import _gate_motion as pose_gate
from strands_robots.tools.serial_tool import COMMAND_ALLOW_ENV as SERIAL_ALLOW
from strands_robots.tools.serial_tool import _gate_write as serial_gate

ROBOT_ALLOW = "STRANDS_ROBOT_COMMAND_ALLOW"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    for env in ("STRANDS_TEST_COMMAND_ALLOW", ROBOT_ALLOW, SERIAL_ALLOW, POSE_ALLOW, UNITREE_ALLOW, ROS_ALLOW):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


def _ctx(response: object) -> MagicMock:
    ctx = MagicMock()
    ctx.interrupt.return_value = response
    return ctx


def _reason(tool: str = "robot", action: str = "set_joint_positions", target: str = "so101") -> dict:
    ctx = _ctx("n")
    gate_motion(tool, action, target, "it moves.", ctx, allow_env="STRANDS_TEST_COMMAND_ALLOW")
    return ctx.interrupt.call_args.kwargs["reason"]


class TestTheReasonCarriesTheRemedy:
    def test_how_to_answer_is_in_the_reason(self) -> None:
        reason = _reason()
        assert reason["how_to_answer"] == how_to_answer("STRANDS_TEST_COMMAND_ALLOW", "so101")
        # the pre-existing fields are untouched: hosts that read them keep working
        assert reason["action"] == "set_joint_positions" and reason["target"] == "so101"
        assert reason["warning"].endswith("Reply 'y' to approve, anything else to deny.")

    def test_the_line_says_paused_how_to_resume_and_what_y_means(self) -> None:
        line = how_to_answer("STRANDS_TEST_COMMAND_ALLOW", "so101")
        assert "paused, not done" in line
        assert "result.interrupts" in line
        assert 'agent([{"interruptResponse": {"interruptId":' in line
        assert '"response": "y"' in line
        assert "anything else denies and nothing moves" in line
        assert "set STRANDS_TEST_COMMAND_ALLOW=so101 instead." in line

    @pytest.mark.parametrize("tool", ["robot", "serial_tool", "pose_tool", "use_unitree", "use_ros"])
    def test_every_tool_on_the_shared_gate_gets_the_same_line(self, tool: str) -> None:
        assert _reason(tool=tool)["how_to_answer"] == how_to_answer("STRANDS_TEST_COMMAND_ALLOW", "so101")

    def test_the_headless_refusal_is_unchanged(self) -> None:
        """No operator, no interrupt: the refusal still names both escape hatches."""
        refusal = gate_motion("robot", "execute", "so101", "it moves.", None, allow_env="STRANDS_TEST_COMMAND_ALLOW")
        assert refusal is not None
        assert "STRANDS_TEST_COMMAND_ALLOW" in refusal and gate_mod.BYPASS_CONSENT_ENV in refusal


class TestTheEntryIsAskedOfTheMatcher:
    """The line names the spelling that pre-approves *this* call, per tool family."""

    def test_an_action_matcher_gets_the_action_and_a_target_matcher_the_target(self) -> None:
        by_action: Callable[[frozenset[str]], bool] = lambda allowed: "move_motor" in allowed  # noqa: E731
        by_target: Callable[[frozenset[str]], bool] = lambda allowed: "/cmd_vel" in allowed  # noqa: E731
        assert allow_entry(by_action, "move_motor", "/cmd_vel") == "move_motor"
        assert allow_entry(by_target, "move_motor", "/cmd_vel") == "/cmd_vel"

    def test_the_action_is_named_when_a_matcher_would_take_either(self) -> None:
        """A tie reads the way the arm tools' own docs do: "pre-approve by action name"."""
        either: Callable[[frozenset[str]], bool] = lambda allowed: bool(allowed & {"move_motor", "/cmd_vel"})  # noqa: E731
        assert allow_entry(either, "move_motor", "/cmd_vel") == "move_motor"

    def test_a_matcher_that_takes_neither_falls_back_to_the_wildcard_every_matcher_honours(self) -> None:
        assert allow_entry(lambda allowed: "*" in allowed, "publish", "/cmd_vel") == "*"


# Every front door onto the shared gate, with the variable it owns: one gated
# call each, made the way the tool makes it.
FRONT_DOORS: list[tuple[str, str, Callable[[ToolContext | None], str | None]]] = [
    (
        "robot",
        ROBOT_ALLOW,
        lambda ctx: gate_motion(
            "robot",
            "set_joint_positions",
            "arm",
            "it moves the arm.",
            ctx,
            allow_env=ROBOT_ALLOW,
            allow_match=lambda allowed: "*" in allowed or "set_joint_positions" in allowed,
        ),
    ),
    (
        "serial_tool",
        SERIAL_ALLOW,
        lambda ctx: serial_gate(
            "feetech_position",
            {"action": "feetech_position", "port": "/dev/ttyACM0", "motor_id": 1, "position": 2048},
            ctx,
        ),
    ),
    (
        "pose_tool",
        POSE_ALLOW,
        lambda ctx: pose_gate(
            "move_motor",
            {"action": "move_motor", "port": "/dev/ttyACM0", "motor_name": "wrist_roll", "position": 10.0},
            ctx,
        ),
    ),
    ("use_unitree", UNITREE_ALLOW, lambda ctx: unitree_gate("sport", "Move", False, ctx)),
    ("use_rosbridge", ROS_ALLOW, lambda ctx: gate_command("publish", "/cmd_vel", ctx, tool="use_rosbridge")),
]


@pytest.mark.parametrize(("tool", "expected_env", "fire"), FRONT_DOORS, ids=[d[0] for d in FRONT_DOORS])
def test_the_line_names_the_variable_and_entry_that_pre_approve_this_call(
    tool: str,
    expected_env: str,
    fire: Callable[[ToolContext | None], str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round trip: set what the line says, and the identical call stops asking.

    A glob (``*_COMMAND_ALLOW``) and a guess at the entry cannot be checked this
    way - and would be wrong here for ``use_unitree`` (``sport.Move``, not
    ``Move``) and for the ROS transports (``/cmd_vel``, not ``publish``, under a
    variable with no transport name in it).
    """
    asked = _ctx("n")
    assert fire(asked) is not None, "a gated call with a 'no' answer is refused"
    line = asked.interrupt.call_args.kwargs["reason"]["how_to_answer"]

    named = re.search(r"set (?P<env>[A-Z0-9_]+)=(?P<entry>\S+) instead\.", line)
    assert named is not None, line
    assert named["env"] == expected_env

    monkeypatch.setenv(named["env"], named["entry"])
    unasked = _ctx("n")
    assert fire(unasked) is None, f"{named['env']}={named['entry']} did not pre-approve the call it was named for"
    assert not unasked.interrupt.called, "a pre-approved call must not ask an operator"


def test_the_real_robot_tool_interrupt_carries_it() -> None:
    """End to end through ``Robot.stream`` on a fake arm: the event a script prints has the line."""
    from strands.types._events import ToolInterruptEvent

    from tests.test_hardware_robot_direct_motion import _make_hw, _robot, _state, _stream

    hw = _make_hw(_robot(calibrated=False))
    events = _stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 95.0})
    assert isinstance(events[-1], ToolInterruptEvent)
    (interrupt,) = events[-1].interrupts
    assert interrupt.reason["how_to_answer"] == how_to_answer(ROBOT_ALLOW, "set_joint_positions")
    # ...and it survives the repr a script prints
    assert "interruptResponse" in repr(interrupt.to_dict())
