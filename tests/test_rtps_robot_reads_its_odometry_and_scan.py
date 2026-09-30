"""An ``RtpsRobot`` given ``odom_topic`` / ``scan_topic`` reads them, typed as a DDS reader needs.

``RtpsRobot`` did not accept ``odom_topic`` or ``scan_topic`` at all (a
``TypeError``), so ``get_pose`` always answered "no odom_topic configured" and
no ``get_pose_<name>`` tool was offered. It also cannot fall back to a live
graph lookup for the type, as the rclpy and rosbridge transports do, so it
defaults the two types to the messages ROS 2 bases publish there.
"""

from __future__ import annotations

from strands_robots.drivers.ros import rtps_robot
from strands_robots.drivers.ros.rtps_robot import RtpsRobot


def test_pose_and_scan_read_their_topics_as_odometry_and_laserscan(monkeypatch):
    seen = []
    monkeypatch.setattr(
        rtps_robot, "rtps_action", lambda **kw: seen.append(kw) or {"status": "success", "content": [{"text": "ok"}]}
    )
    base = RtpsRobot("base", "/cmd_vel", odom_topic="/odom", scan_topic="/scan")
    assert base.get_pose()["status"] == "success"
    assert base.get_scan()["status"] == "success"
    assert [(kw["action"], kw["topic"], kw["type"]) for kw in seen] == [
        ("echo", "/odom", "nav_msgs/msg/Odometry"),
        ("echo", "/scan", "sensor_msgs/msg/LaserScan"),
    ]


def test_the_pose_tool_is_offered_only_with_an_odom_topic():
    names = lambda robot: {t.tool_name for t in robot.tools}  # noqa: E731
    assert "get_pose_base" in names(RtpsRobot("base", "/cmd_vel", odom_topic="/odom"))
    assert "get_pose_base" not in names(RtpsRobot("base", "/cmd_vel"))


def test_an_explicit_type_wins(monkeypatch):
    seen = []
    monkeypatch.setattr(rtps_robot, "rtps_action", lambda **kw: seen.append(kw) or {"status": "success", "content": []})
    RtpsRobot("base", "/cmd_vel", odom_topic="/pose", odom_type="geometry_msgs/msg/PoseStamped").get_pose()
    assert seen[0]["type"] == "geometry_msgs/msg/PoseStamped"
