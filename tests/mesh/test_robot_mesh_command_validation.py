"""Input-validation contract for the ``robot_mesh`` dispatcher.

``robot_mesh`` validates the ``broadcast`` / ``send`` / ``tell`` command
bodies *before* any mesh dispatch or human-in-the-loop approval gate, and
always returns a structured ``{"status": "error", ...}`` dict (it never
raises out of the dispatcher). These tests pin that fail-closed contract so
a malformed agent tool call is rejected with an actionable message instead
of reaching the transport.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.tools.robot_mesh import _reset_rate_limits, robot_mesh


def _call(**kwargs):
    """Invoke the underlying tool fn (Strands ``@tool`` wraps it as ``.original``)."""
    fn = getattr(robot_mesh, "original", robot_mesh)
    return fn(**kwargs)


def _text(out) -> str:
    return out["content"][0]["text"]


@pytest.fixture(autouse=True)
def _isolate_rate_limits():
    """Each validation case consumes a per-action rate-limit slot; reset so
    the cases stay independent of collection order."""
    _reset_rate_limits()
    yield
    _reset_rate_limits()


def test_broadcast_without_command_is_rejected():
    out = _call(action="broadcast", command="", tool_context=MagicMock())
    assert out["status"] == "error"
    assert "broadcast requires command" in _text(out)


def test_broadcast_with_malformed_json_is_rejected():
    out = _call(action="broadcast", command="{not valid", tool_context=MagicMock())
    assert out["status"] == "error"
    assert "not valid JSON" in _text(out)


def test_broadcast_with_non_object_json_is_rejected():
    # A JSON array parses but is not a command object -> reject.
    out = _call(action="broadcast", command="[1, 2, 3]", tool_context=MagicMock())
    assert out["status"] == "error"
    assert "JSON object" in _text(out)


def test_broadcast_with_disallowed_action_is_rejected():
    out = _call(action="broadcast", command='{"action": "rm_rf"}', tool_context=MagicMock())
    assert out["status"] == "error"
    assert "broadcast rejected" in _text(out)


def test_send_without_target_is_rejected():
    out = _call(action="send", target="", command='{"action": "stop"}', tool_context=MagicMock())
    assert out["status"] == "error"
    assert "send requires target" in _text(out)


def test_tell_with_overlong_instruction_is_rejected():
    # policy_port is forwarded into the synthesised execute command; pass one
    # so the rejection path covers the port-forwarding branch too. The
    # instruction-length check still fires first, so the command is rejected.
    out = _call(
        action="tell",
        target="peer-b",
        instruction="x" * 5000,
        policy_port=5555,
        tool_context=MagicMock(),
    )
    assert out["status"] == "error"
    assert "tell rejected" in _text(out)


def test_interrupt_gated_action_without_tool_context_fails_closed():
    # emergency_stop is HITL-gated by default; with no tool_context the
    # dispatcher must refuse rather than fire a fleet-wide stop unattended.
    out = _call(action="emergency_stop", tool_context=None)
    assert out["status"] == "error"
    assert "human-in-the-loop interrupt" in _text(out)
