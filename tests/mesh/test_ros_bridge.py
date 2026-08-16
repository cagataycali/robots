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
    """Every unconditional capability is exposed, suffixed with the node name.

    ``stop`` is in this set because a drive without a ``duration`` latches:
    this assertion previously recorded its absence, which is what kept the
    missing halt tool invisible.
    """
    names = {t.tool_name for t in _turtle().tools}
    assert names == {"drive_turtlesim", "stop_turtlesim", "get_pose_turtlesim"}


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


# navigate_to -----------------------------------------------------------------


def _nav_turtle() -> RosBridgedRobot:
    return RosBridgedRobot.from_ros(
        node_name="tb4",
        cmd_vel_topic="/cmd_vel",
        odom_topic="/odom",
        scan_topic="/scan",
        nav_action="/navigate_to_pose",
    )


def test_navigate_to_forwards_action_goal(rec: _Recorder) -> None:
    import math

    _nav_turtle().navigate_to(x=1.0, y=2.0, yaw=math.pi / 2, timeout=60.0)
    call = rec.calls[0]
    assert call["action"] == "action_send_goal"
    assert call["action_name"] == "/navigate_to_pose"
    assert call["type"] == "nav2_msgs/action/NavigateToPose"
    assert call["timeout"] == 60.0
    pose = call["fields"]["pose"]
    assert pose["header"]["frame_id"] == "map"
    assert pose["pose"]["position"] == {"x": 1.0, "y": 2.0}
    # yaw=pi/2 -> planar quaternion (z=sin(pi/4), w=cos(pi/4)).
    assert pose["pose"]["orientation"]["z"] == pytest.approx(math.sin(math.pi / 4))
    assert pose["pose"]["orientation"]["w"] == pytest.approx(math.cos(math.pi / 4))


def test_navigate_to_without_nav_action_is_error(rec: _Recorder) -> None:
    result = _turtle().navigate_to(x=1.0, y=1.0)
    assert result["status"] == "error"
    assert "no nav_action configured" in result["content"][0]["text"]
    assert rec.calls == []  # nothing forwarded


def test_invalid_nav_action_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid nav_action"):
        RosBridgedRobot("tb", "/cmd_vel", "/odom", nav_action="/nav goal")


def test_navigate_tool_exposed_only_when_configured() -> None:
    with_nav = {t.tool_name for t in _nav_turtle().tools}
    without_nav = {t.tool_name for t in _turtle().tools}
    assert "navigate_tb4" in with_nav
    assert not any(name.startswith("navigate_") for name in without_nav)


def test_navigate_tool_forwards_to_navigate_to(rec: _Recorder) -> None:
    robot = _nav_turtle()
    nav_tool: Any = next(t for t in robot.tools if t.tool_name == "navigate_tb4")
    nav_tool(x=0.5, y=-0.5)
    assert rec.calls and rec.calls[0]["action"] == "action_send_goal"
