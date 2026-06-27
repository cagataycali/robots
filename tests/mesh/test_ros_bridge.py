"""Behavior tests for the ROS 2 mesh bridge (:class:`RosBridgedRobot`).

The bridge owns no ROS 2 state - every method forwards to ``use_ros``. These
tests patch the forwarded ``use_ros`` symbol so they run with NO ROS 2 present,
asserting the bridge builds the right ``use_ros`` calls (correct topic, type,
Twist field mapping, message count) and exposes correctly-named agent tools.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.mesh.ros_bridge as bridge_mod
from strands_robots.mesh import RosBridgedRobot


class _Recorder:
    """Records the kwargs of each forwarded ``use_ros`` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(bridge_mod, "use_ros", recorder)
    return recorder


def _turtle() -> RosBridgedRobot:
    return RosBridgedRobot.from_ros(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
        odom_topic="/turtle1/pose",
        odom_type="turtlesim/msg/Pose",
    )


def test_from_ros_matches_constructor() -> None:
    a = RosBridgedRobot("tb", "/cmd_vel", "/odom")
    b = RosBridgedRobot.from_ros("tb", "/cmd_vel", "/odom")
    assert a.cmd_vel_topic == b.cmd_vel_topic == "/cmd_vel"
    assert a.odom_topic == b.odom_topic == "/odom"
    assert a.scan_topic is None and b.scan_topic is None


@pytest.mark.parametrize(
    "node_name,cmd_vel_topic,odom_topic",
    [
        ("bad name", "/cmd_vel", "/odom"),
        ("tb", "/has;semicolon", "/odom"),
        ("tb", "/cmd_vel", ""),
    ],
)
def test_invalid_names_rejected_at_construction(node_name: str, cmd_vel_topic: str, odom_topic: str) -> None:
    with pytest.raises(ValueError):
        RosBridgedRobot(node_name, cmd_vel_topic, odom_topic)


def test_drive_publishes_twist(rec: _Recorder) -> None:
    _turtle().drive(linear=2.0, angular=1.5)
    (call,) = rec.calls
    assert call["action"] == "publish"
    assert call["topic"] == "/turtle1/cmd_vel"
    assert call["type"] == "geometry_msgs/msg/Twist"
    assert call["fields"] == {"linear": {"x": 2.0}, "angular": {"z": 1.5}}
    assert call["count"] == 1


def test_drive_duration_sets_message_count(rec: _Recorder) -> None:
    # publish_rate defaults to 10 Hz -> 1.5 s == 15 messages.
    _turtle().drive(linear=1.0, duration=1.5)
    assert rec.calls[0]["count"] == 15


def test_stop_publishes_zero_velocity(rec: _Recorder) -> None:
    _turtle().stop()
    assert rec.calls[0]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def test_get_pose_echoes_odom_topic(rec: _Recorder) -> None:
    _turtle().get_pose()
    (call,) = rec.calls
    assert call["action"] == "echo"
    assert call["topic"] == "/turtle1/pose"
    assert call["type"] == "turtlesim/msg/Pose"


def test_get_scan_without_topic_returns_error(rec: _Recorder) -> None:
    result = _turtle().get_scan()
    assert result["status"] == "error"
    assert rec.calls == []  # nothing forwarded


def test_get_scan_with_topic_echoes(rec: _Recorder) -> None:
    robot = RosBridgedRobot("tb", "/cmd_vel", "/odom", scan_topic="/scan")
    robot.get_scan()
    assert rec.calls[0]["action"] == "echo"
    assert rec.calls[0]["topic"] == "/scan"


def test_tools_are_named_per_instance() -> None:
    names = {t.tool_name for t in _turtle().tools}
    assert names == {"drive_turtlesim", "get_pose_turtlesim"}


def test_tools_include_scan_only_when_configured() -> None:
    with_scan = RosBridgedRobot("tb", "/cmd_vel", "/odom", scan_topic="/scan")
    names = {t.tool_name for t in with_scan.tools}
    assert "get_scan_tb" in names


def test_drive_tool_forwards_to_instance(rec: _Recorder) -> None:
    tools = {t.tool_name: t for t in _turtle().tools}
    drive_tool: Any = tools["drive_turtlesim"]
    drive_tool(linear=1.0)
    assert rec.calls[0]["action"] == "publish"
    assert rec.calls[0]["fields"]["linear"]["x"] == 1.0
