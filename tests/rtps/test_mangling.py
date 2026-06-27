"""Unit tests for ROS 2 <-> DDS name mangling (pure string transforms)."""

from __future__ import annotations

import pytest

from strands_robots.rtps.mangling import dds_topic_name, dds_type_name, ros_topic_name


@pytest.mark.parametrize(
    ("ros", "dds"),
    [
        ("/turtle1/cmd_vel", "rt/turtle1/cmd_vel"),
        ("/cmd_vel", "rt/cmd_vel"),
        ("/a/b/c", "rt/a/b/c"),
    ],
)
def test_topic_roundtrip(ros: str, dds: str) -> None:
    assert dds_topic_name(ros) == dds
    assert ros_topic_name(dds) == ros


@pytest.mark.parametrize("bad", ["cmd_vel", "", "/", "/bad name", "/a/", "/x;y"])
def test_topic_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 topic"):
        dds_topic_name(bad)


def test_ros_topic_name_requires_prefix() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        ros_topic_name("/turtle1/cmd_vel")  # missing the rt prefix


@pytest.mark.parametrize(
    ("ros", "dds"),
    [
        ("geometry_msgs/msg/Twist", "geometry_msgs::msg::dds_::Twist_"),
        ("sensor_msgs/msg/LaserScan", "sensor_msgs::msg::dds_::LaserScan_"),
        ("turtlesim/msg/Pose", "turtlesim::msg::dds_::Pose_"),
    ],
)
def test_type_mangling(ros: str, dds: str) -> None:
    assert dds_type_name(ros) == dds


@pytest.mark.parametrize("bad", ["Twist", "geometry_msgs/Twist", "a/b/c/d", "pkg/badkind/Name"])
def test_type_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 type"):
        dds_type_name(bad)
