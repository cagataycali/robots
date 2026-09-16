"""The gate's interrupt tells a headless script how to answer it.

``agent("rotate the wrist 5 degrees")`` on a real arm returns a PAUSED result:
the operator gate raised an interrupt and the SDK handed it back for someone to
answer. A script that does ``print(agent(...))`` sees the interrupt list's repr
- the action, the target, the warning - and nothing that says the run is paused
or how to resume it. The ``reason`` dict is the one thing that script is
certainly printing, so it now carries ``how_to_answer``: the exact
``agent([{"interruptResponse": ...}])`` form, what ``y`` means, and the
pre-approval variable for a script with no operator.

The pre-approval half is graded through the real call sites rather than a
sentence, because which spelling the allowlist accepts differs per tool -
``serial_tool`` and ``pose_tool`` match the action, ``use_unitree`` the
``service.operation`` pair, the ROS transports the surface name. The value the
interrupt names has to be one that would let the identical command through, so
each row here sets it and re-runs the gate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.hardware_robot import COMMAND_ALLOW_ENV as ROBOT_ALLOW_ENV
from strands_robots.tools import _command_gate as gate_mod
from strands_robots.tools._command_gate import gate_command, gate_motion, how_to_answer
from strands_robots.tools.g1.use_unitree import COMMAND_ALLOW_ENV as UNITREE_ALLOW_ENV
from strands_robots.tools.g1.use_unitree import _gate as unitree_gate
from strands_robots.tools.pose_tool import COMMAND_ALLOW_ENV as POSE_ALLOW_ENV
from strands_robots.tools.pose_tool import _gate_motion as pose_gate
from strands_robots.tools.serial_tool import COMMAND_ALLOW_ENV as SERIAL_ALLOW_ENV
from strands_robots.tools.serial_tool import _gate_write as serial_gate

TEST_ALLOW_ENV = "STRANDS_TEST_COMMAND_ALLOW"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No bypass, nothing pre-approved, and the audit row lands in the tmp dir."""
    for name in (
        gate_mod.BYPASS_CONSENT_ENV,
        TEST_ALLOW_ENV,
        ROBOT_ALLOW_ENV,
        gate_mod.COMMAND_ALLOW_ENV,
        SERIAL_ALLOW_ENV,
        POSE_ALLOW_ENV,
        UNITREE_ALLOW_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


def _ctx(response: object = "n") -> MagicMock:
    ctx = MagicMock()
    ctx.interrupt.return_value = response
    return ctx


def _asked(ctx: MagicMock) -> dict[str, Any]:
    """The ``reason`` the gate handed the operator."""
    return dict(ctx.interrupt.call_args.kwargs["reason"])


# Each row calls a real gate site with its own matcher, so the accepted spelling
# is the tool's, not this file's: (label, run the gate, allowlist variable, the
# value that pre-approves this command).
SITES = [
    (
        "serial_tool",
        lambda ctx: serial_gate("feetech_position", {"port": "/dev/ttyACM0", "position": 512}, ctx),
        SERIAL_ALLOW_ENV,
        "feetech_position",
    ),
    (
        "pose_tool",
        lambda ctx: pose_gate("move_motor", {"port": "/dev/ttyACM0", "motor_name": "elbow"}, ctx),
        POSE_ALLOW_ENV,
        "move_motor",
    ),
    (
        "use_unitree",
        lambda ctx: unitree_gate("loco", "SetVelocity", True, ctx),
        UNITREE_ALLOW_ENV,
        "loco.SetVelocity",
    ),
    (
        "use_ros",
        lambda ctx: gate_command("publish", "/cmd_vel", ctx, tool="use_ros"),
        gate_mod.COMMAND_ALLOW_ENV,
        "/cmd_vel",
    ),
    (
        "use_rosbridge",
        lambda ctx: gate_command("service_call", "/motor_enable", ctx, tool="use_rosbridge"),
        gate_mod.COMMAND_ALLOW_ENV,
        "/motor_enable",
    ),
]
IDS = [row[0] for row in SITES]


class TestTheReasonCarriesTheRemedy:
    def test_how_to_answer_is_in_the_reason(self) -> None:
        ctx = _ctx()
        gate_motion("robot", "execute", "so101", "it moves.", ctx, allow_env=TEST_ALLOW_ENV)
        reason = _asked(ctx)

        assert "how_to_answer" in reason
        # the pre-existing fields are untouched: hosts that read them keep working
        assert reason["action"] == "execute" and reason["target"] == "so101"
        assert reason["warning"].endswith("Reply 'y' to approve, anything else to deny.")

    def test_the_line_says_paused_how_to_resume_and_what_y_means(self) -> None:
        ctx = _ctx()
        gate_motion("robot", "execute", "so101", "it moves.", ctx, allow_env=TEST_ALLOW_ENV)
        line = _asked(ctx)["how_to_answer"]

        assert "paused, not done" in line
        assert "result.interrupts" in line
        assert 'agent([{"interruptResponse": {"interruptId":' in line
        assert '"response": "y"' in line
        assert "anything else denies and nothing moves" in line

    def test_the_headless_refusal_is_unchanged(self) -> None:
        """No operator, no interrupt: the refusal still names both escape hatches."""
        refusal = gate_motion("robot", "execute", "so101", "it moves.", None, allow_env=TEST_ALLOW_ENV)

        assert refusal is not None
        assert TEST_ALLOW_ENV in refusal and gate_mod.BYPASS_CONSENT_ENV in refusal


class TestThePreapprovalItNamesIsOneTheGateAccepts:
    """The value in the line has to lift the gate on the identical command."""

    @pytest.mark.parametrize(("label", "run_gate", "env", "accepted"), SITES, ids=IDS)
    def test_the_line_names_the_spelling_this_tool_matches(
        self, label: str, run_gate: Any, env: str, accepted: str
    ) -> None:
        ctx = _ctx()
        run_gate(ctx)

        assert f"pre-approve this command instead with {env}={accepted}." in _asked(ctx)["how_to_answer"]

    @pytest.mark.parametrize(("label", "run_gate", "env", "accepted"), SITES, ids=IDS)
    def test_setting_what_the_line_says_lets_the_same_command_through(
        self, label: str, run_gate: Any, env: str, accepted: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round trip: read the value out of the line, set it, and the gate no longer asks."""
        first = _ctx()
        run_gate(first)
        hint = _asked(first)["how_to_answer"].rsplit("pre-approve this command instead with ", 1)[1].rstrip(".")
        variable, _, value = hint.partition("=")
        monkeypatch.setenv(variable, value)

        second = _ctx()
        assert run_gate(second) is None, f"{label}: {hint} did not pre-approve the command it was offered for"
        assert not second.interrupt.called, f"{label}: still asked an operator after {hint}"

    def test_the_action_name_is_not_the_answer_for_every_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Why the line is derived: a shared sentence naming the action is wrong for these two.

        ``use_unitree`` matches the ``service.operation`` pair and the ROS
        transports match the surface, so pre-approving ``SetVelocity`` or
        ``publish`` leaves the gate exactly where it was.
        """
        for action, run_gate, env in (
            (
                "SetVelocity",
                lambda ctx: unitree_gate("loco", "SetVelocity", True, ctx),
                UNITREE_ALLOW_ENV,
            ),
            (
                "publish",
                lambda ctx: gate_command("publish", "/cmd_vel", ctx, tool="use_ros"),
                gate_mod.COMMAND_ALLOW_ENV,
            ),
        ):
            monkeypatch.setenv(env, action)
            ctx = _ctx()
            run_gate(ctx)
            assert ctx.interrupt.called, f"{env}={action} lifted the gate; the table's row is stale"
            monkeypatch.delenv(env)

    def test_a_matcher_that_accepts_neither_names_the_variable_alone(self) -> None:
        """No value is offered when the caller's matcher would accept none: no invented advice."""
        ctx = _ctx()
        gate_motion(
            "odd_tool",
            "publish",
            "/cmd_vel",
            "it moves.",
            ctx,
            allow_env=TEST_ALLOW_ENV,
            allow_match=lambda allowed: False,
        )

        line = _asked(ctx)["how_to_answer"]
        assert line.endswith(f"pre-approve this command instead with {TEST_ALLOW_ENV}.")


def test_the_real_robot_tool_interrupt_carries_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """End to end through ``Robot.stream`` on a fake arm: the event a script prints has the line."""
    from strands.types._events import ToolInterruptEvent

    from tests.test_hardware_robot_stream_gates_real_dispatch import _make_robot, _state, _stream

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    hw = _make_robot()
    try:
        events = _stream(hw, "execute", _state(None))
    finally:
        hw.cleanup()

    assert isinstance(events[-1], ToolInterruptEvent), events
    (interrupt,) = events[-1].interrupts
    line = interrupt.reason["how_to_answer"]
    assert "paused, not done" in line
    assert f"{ROBOT_ALLOW_ENV}=execute." in line
    # ...and it survives the repr a script prints
    assert "interruptResponse" in repr(interrupt.to_dict())


def test_how_to_answer_asks_the_matcher_rather_than_assuming() -> None:
    """The helper offers the first of action/target the caller's matcher accepts."""
    assert how_to_answer("publish", "/cmd_vel", "VAR", lambda allowed: "/cmd_vel" in allowed).endswith(
        "with VAR=/cmd_vel."
    )
    assert how_to_answer("publish", "/cmd_vel", "VAR", lambda allowed: "publish" in allowed).endswith(
        "with VAR=publish."
    )
