"""Tests for strands_robots.tools.robot_mesh — agent-facing dispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import robot_mesh


def _strands_call(**kwargs):
    """Strands @tool wraps the function — invoke via .original."""
    fn = getattr(robot_mesh, "original", None)
    if fn is None:
        return robot_mesh(**kwargs)
    return fn(**kwargs)


@pytest.fixture
def fake_local_mesh():
    """Patch get_local_robots() to return a single fake mesh keyed by peer."""
    fake = MagicMock(name="LocalMesh")
    fake.peer_id = "local-a"
    fake.peer_type = "sim"
    fake.inbox = {}
    with (
        patch(
            "strands_robots.mesh.get_local_robots",
            return_value={"local-a": fake},
        ),
        patch("strands_robots.mesh_session.get_peers", return_value=[]),
    ):
        yield fake


@pytest.fixture
def fake_no_local():
    """Patch get_local_robots()/get_peers() to return empty."""
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={}),
        patch("strands_robots.mesh_session.get_peers", return_value=[]),
    ):
        yield


def test_peers_lists_local_and_remote(fake_local_mesh):
    with patch(
        "strands_robots.mesh_session.get_peers",
        return_value=[{"peer_id": "remote-1", "type": "robot", "hostname": "host1", "age": 3}],
    ):
        out = _strands_call(action="peers")
    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "local-a" in text
    assert "remote-1" in text


def test_peers_no_local_no_remote(fake_no_local):
    out = _strands_call(action="peers")
    assert out["status"] == "success"
    assert "No peers" in out["content"][0]["text"]


def test_status_returns_counts(fake_local_mesh):
    out = _strands_call(action="status")
    assert out["status"] == "success"
    assert "local=1" in out["content"][0]["text"]


def test_tell_requires_target_and_instruction(fake_local_mesh):
    out = _strands_call(action="tell")
    assert out["status"] == "error"


def test_tell_invokes_mesh_tell(fake_local_mesh):
    fake_local_mesh.tell.return_value = {"executed": "go"}
    out = _strands_call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "success"
    fake_local_mesh.tell.assert_called_once()
    args = fake_local_mesh.tell.call_args
    assert args.args == ("peer-b", "go")


def test_send_requires_command(fake_local_mesh):
    out = _strands_call(action="send", target="peer-b")
    assert out["status"] == "error"
    assert "command" in out["content"][0]["text"].lower()


def test_send_rejects_invalid_json(fake_local_mesh):
    out = _strands_call(action="send", target="peer-b", command="not json")
    assert out["status"] == "error"
    assert "JSON" in out["content"][0]["text"]


def test_send_invokes_mesh_send(fake_local_mesh):
    fake_local_mesh.send.return_value = {"ok": 1}
    out = _strands_call(
        action="send",
        target="peer-b",
        command='{"action": "status"}',
        timeout=5.0,
    )
    assert out["status"] == "success"
    args = fake_local_mesh.send.call_args
    assert args.args[0] == "peer-b"
    assert args.args[1] == {"action": "status"}
    assert args.kwargs["timeout"] == 5.0


def test_broadcast_invokes_mesh_broadcast(fake_local_mesh):
    fake_local_mesh.broadcast.return_value = [{"a": 1}, {"b": 2}]
    out = _strands_call(action="broadcast", command='{"action":"status"}')
    assert out["status"] == "success"
    assert "2 responses" in out["content"][0]["text"]


def test_stop_requires_target(fake_local_mesh):
    out = _strands_call(action="stop")
    assert out["status"] == "error"


def test_stop_sends_stop_action(fake_local_mesh):
    fake_local_mesh.send.return_value = {"stopped": True}
    _strands_call(action="stop", target="peer-b")
    args = fake_local_mesh.send.call_args
    assert args.args[1] == {"action": "stop"}


def test_emergency_stop_invokes_mesh_emergency_stop(fake_local_mesh):
    fake_local_mesh.emergency_stop.return_value = [{"a": 1}, {"b": 2}]
    out = _strands_call(action="emergency_stop")
    assert out["status"] == "success"
    fake_local_mesh.emergency_stop.assert_called_once()
    assert "2 responses" in out["content"][0]["text"]


def test_subscribe_requires_target(fake_local_mesh):
    out = _strands_call(action="subscribe")
    assert out["status"] == "error"


def test_subscribe_calls_mesh_subscribe(fake_local_mesh):
    fake_local_mesh.subscribe.return_value = "topic-name"
    out = _strands_call(action="subscribe", target="reachy/*", name="reachy")
    assert out["status"] == "success"
    fake_local_mesh.subscribe.assert_called_once()


def test_watch_requires_target(fake_local_mesh):
    out = _strands_call(action="watch")
    assert out["status"] == "error"


def test_watch_calls_on_stream(fake_local_mesh):
    fake_local_mesh.on_stream.return_value = "stream:peer-b"
    out = _strands_call(action="watch", target="peer-b")
    assert out["status"] == "success"
    fake_local_mesh.on_stream.assert_called_once_with("peer-b")


def test_inbox_returns_buffered_messages(fake_local_mesh):
    fake_local_mesh.inbox = {"sub-a": [("topic", {"x": 1}), ("topic", {"x": 2})]}
    out = _strands_call(action="inbox", name="sub-a")
    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "2 total" in text


def test_inbox_with_no_messages(fake_local_mesh):
    out = _strands_call(action="inbox", name="empty")
    assert out["status"] == "success"
    assert "no messages" in out["content"][0]["text"]


def test_unknown_action_returns_error(fake_local_mesh):
    out = _strands_call(action="warp")
    assert out["status"] == "error"
    assert "unknown action" in out["content"][0]["text"]


def test_actions_without_local_mesh_fail(fake_no_local):
    out = _strands_call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    assert "no local mesh" in out["content"][0]["text"]
