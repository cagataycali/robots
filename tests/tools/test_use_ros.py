"""Behavior tests for the ``use_ros`` agent tool.

The tool bridges a Strands agent to a ROS 2 graph **entirely in-process through
``rclpy``** - there is no ``ros2`` CLI shelling and no generated-code snippets.
These tests run with NO ROS 2 installed: the rclpy-facing helpers
(``_list_topics`` / ``_echo`` / ``_publish`` / ``_service_call`` / ...) and the
backend-availability probe are monkeypatched, so every action-dispatch branch,
the agent-input validation, the no-backend error path, and the structured
error-return contract are exercised hardware- and ROS-free.

It also pins package-wide contracts:

* No emoji / non-ASCII in any returned ``text``.
* ``fields`` payloads (bool / None / nested) are passed straight through to the
  rclpy helper as a real Python dict - never serialised into source - so types
  are preserved by construction.
* Backend errors surface as a structured ``{"status": "error"}`` result, never
  a raised exception.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.tools.use_ros as ros_mod

# Reference the tool via a module-local alias rather than a second `from`
# import: the tests monkeypatch module internals through `ros_mod`, so the
# module object is the single source of truth and a dual import is avoided.
use_ros = ros_mod.use_ros


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _ascii_only(result: dict[str, Any]) -> None:
    text = _texts(result)
    assert text.isascii(), f"non-ASCII in tool output: {text!r}"


@pytest.fixture(autouse=True)
def _backend_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a present rclpy backend; opt out where needed."""
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)


# Validation ----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/foo; rm -rf", "/a b", "/x|y", "../etc", "/a$(x)"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_ros(action="echo", topic=bad)
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)
    _ascii_only(result)


def test_invalid_type_rejected() -> None:
    result = use_ros(action="publish", topic="/cmd_vel", type="not_a_type")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


def test_invalid_service_rejected() -> None:
    result = use_ros(action="service_call", service="/spawn bad", type="turtlesim/srv/Spawn")
    assert result["status"] == "error"
    assert "invalid service" in _texts(result)


# Status --------------------------------------------------------------------


def test_status_reports_rclpy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
    result = use_ros(action="status")
    assert result["status"] == "success"
    assert "backend: rclpy (in-process)" in _texts(result)
    _ascii_only(result)


def test_status_reports_none_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)
    result = use_ros(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)
    assert "ROS 2" in _texts(result)
    _ascii_only(result)


# Listings ------------------------------------------------------------------


def test_list_topics_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_list_topics", lambda: "/turtle1/cmd_vel [geometry_msgs/msg/Twist]")
    result = use_ros(action="list_topics")
    assert result["status"] == "success"
    assert "/turtle1/cmd_vel" in _texts(result)
    _ascii_only(result)


def test_list_nodes_and_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_list_nodes", lambda: "/turtlesim")
    monkeypatch.setattr(ros_mod, "_list_services", lambda: "/spawn [turtlesim/srv/Spawn]")
    assert "/turtlesim" in _texts(use_ros(action="list_nodes"))
    assert "/spawn" in _texts(use_ros(action="list_services"))


# info ----------------------------------------------------------------------


def test_info_requires_target() -> None:
    result = use_ros(action="info")
    assert result["status"] == "error"
    assert "requires topic or service" in _texts(result)


def test_info_returns_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_info", lambda target: f"topic info {target}:\n  publishers: 1")
    result = use_ros(action="info", topic="/turtle1/pose")
    assert result["status"] == "success"
    assert "topic info /turtle1/pose" in _texts(result)


def test_info_miss_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_info", lambda target: None)
    result = use_ros(action="info", topic="/nope")
    assert result["status"] == "error"
    assert "no info for /nope" in _texts(result)


# echo ----------------------------------------------------------------------


def test_echo_requires_topic() -> None:
    assert use_ros(action="echo")["status"] == "error"


def test_echo_autoresolves_type_and_returns_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_resolve_topic_type", lambda topic: "turtlesim/msg/Pose")
    samples = [{"x": 5.5, "y": 1.0}, {"x": 6.0, "y": 1.0}]

    def fake_echo(topic: str, msg_type: str, timeout: float, count: int) -> list[dict[str, Any]]:
        assert msg_type == "turtlesim/msg/Pose"  # auto-resolved type reached the helper
        return samples

    monkeypatch.setattr(ros_mod, "_echo", fake_echo)
    result = use_ros(action="echo", topic="/turtle1/pose", count=2)
    assert result["status"] == "success"
    assert "turtlesim/msg/Pose" in _texts(result)
    assert "5.5" in _texts(result)
    _ascii_only(result)


def test_echo_unresolvable_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_resolve_topic_type", lambda topic: None)
    result = use_ros(action="echo", topic="/turtle1/pose")
    assert result["status"] == "error"
    assert "cannot resolve type" in _texts(result)


# publish / service_call ----------------------------------------------------


def test_publish_requires_topic_and_type() -> None:
    assert use_ros(action="publish", topic="/cmd_vel")["status"] == "error"


def test_publish_dispatches_with_real_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_publish(topic, msg_type, fields, count, rate) -> None:
        captured.update(topic=topic, msg_type=msg_type, fields=fields, count=count)

    monkeypatch.setattr(ros_mod, "_publish", fake_publish)
    result = use_ros(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "enabled": True, "tag": None},
        count=3,
    )
    assert result["status"] == "success"
    assert "published 3 message(s) to /turtle1/cmd_vel" in _texts(result)
    # The payload reaches the rclpy helper as a real Python dict with types intact.
    assert captured["fields"] == {"linear": {"x": 2.0}, "enabled": True, "tag": None}


def test_service_call_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_service_call", lambda service, srv_type, fields, timeout: {"name": "t2"})
    result = use_ros(
        action="service_call",
        service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"},
    )
    assert result["status"] == "success"
    assert "t2" in _texts(result)


# Error / no-backend contracts ----------------------------------------------


def test_no_backend_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)
    result = use_ros(action="list_topics")
    assert result["status"] == "error"
    assert "ROS 2" in _texts(result) and "rclpy" in _texts(result)
    _ascii_only(result)


def test_timeout_surfaces_as_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise TimeoutError("service /spawn not available within 5.0s")

    monkeypatch.setattr(ros_mod, "_service_call", boom)
    result = use_ros(action="service_call", service="/spawn", type="turtlesim/srv/Spawn")
    assert result["status"] == "error"
    assert "not available" in _texts(result)


def test_type_resolution_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise KeyError("geometry_msgs/msg/Nope")

    monkeypatch.setattr(ros_mod, "_publish", boom)
    result = use_ros(action="publish", topic="/cmd", type="geometry_msgs/msg/Nope")
    assert result["status"] == "error"
    assert "publish failed" in _texts(result)


def test_unknown_action_errors() -> None:
    result = use_ros(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)
