"""The RTPS bundle carries the messages a mobile base and its operator exchange, laid out as ROS 2 lays them out.

``docs/learn/ros2.md`` opens with an agent that echoes ``/odom`` over
``use_rtps``, and ``RtpsRobot.get_pose`` / ``get_scan`` read ``/odom`` and
``/scan`` - but the bundle had no ``nav_msgs/msg/Odometry``, no
``sensor_msgs/msg/LaserScan`` and not even ``std_msgs/msg/String``, so each was
refused as "not in the RTPS IDL bundle". Each layout below is the output of
``ros2 interface show --no-comments <type>`` on Jazzy; a sample whose field order
or kinds differ is dropped by a real node without an error on either side, so
the order is what is pinned. ``tests_integ/tools/test_use_rtps_bundle_interop_live.py``
checks the same types on the wire against a real ROS 2 node, both directions.
"""

from __future__ import annotations

import dataclasses
import typing

import pytest

import strands_robots.rtps.idl as idl

pytestmark = pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")

#: ``ros2 interface show --no-comments`` on ROS 2 Jazzy, top-level fields only.
JAZZY = {
    "std_msgs/msg/String": "string data",
    "std_msgs/msg/Bool": "bool data",
    "std_msgs/msg/Int32": "int32 data",
    "std_msgs/msg/Float32": "float32 data",
    "std_msgs/msg/Float64": "float64 data",
    "geometry_msgs/msg/PoseStamped": "std_msgs/Header header; Pose pose",
    "geometry_msgs/msg/TwistStamped": "std_msgs/Header header; Twist twist",
    "geometry_msgs/msg/PoseWithCovariance": "Pose pose; float64[36] covariance",
    "geometry_msgs/msg/TwistWithCovariance": "Twist twist; float64[36] covariance",
    "nav_msgs/msg/Odometry": (
        "std_msgs/Header header; string child_frame_id; "
        "geometry_msgs/PoseWithCovariance pose; geometry_msgs/TwistWithCovariance twist"
    ),
    "sensor_msgs/msg/LaserScan": (
        "std_msgs/Header header; float32 angle_min; float32 angle_max; float32 angle_increment; "
        "float32 time_increment; float32 scan_time; float32 range_min; float32 range_max; "
        "float32[] ranges; float32[] intensities"
    ),
    "sensor_msgs/msg/Imu": (
        "std_msgs/Header header; geometry_msgs/Quaternion orientation; float64[9] orientation_covariance; "
        "geometry_msgs/Vector3 angular_velocity; float64[9] angular_velocity_covariance; "
        "geometry_msgs/Vector3 linear_acceleration; float64[9] linear_acceleration_covariance"
    ),
}


def _kind(hint: typing.Any) -> str:
    """The .msg spelling of a resolved cyclonedds field annotation."""
    if dataclasses.is_dataclass(hint):
        return idl_to_ros(hint)
    text = repr(hint)
    if hint is str:
        return "string"
    if hint is bool:
        return "bool"
    for scalar in ("float64", "float32", "int32", "uint32", "uint8"):
        if f"'{scalar}'" in text or text.endswith(scalar) or f".{scalar}" in text or f"{scalar}]" in text:
            base = scalar
            break
    else:
        base = "string" if "str" in text else text
    if "array" in text:
        size = [p for p in text.replace("]", ",").replace(")", ",").split(",") if p.strip().isdigit()]
        return f"{base}[{size[-1].strip()}]"
    if "sequence" in text:
        return f"{base}[]"
    return base


def idl_to_ros(cls: typing.Any) -> str:
    return cls.__idl_typename__.split("::")[-1].rstrip("_")


def _layout(cls: typing.Any) -> list[tuple[str, str]]:
    hints = typing.get_type_hints(cls, include_extras=True)
    return [(f.name, _kind(hints[f.name])) for f in dataclasses.fields(cls)]


def _expected(ros_type: str) -> list[tuple[str, str]]:
    out = []
    for decl in JAZZY[ros_type].split("; "):
        kind, name = decl.split(" ")
        out.append((name, kind.split("/")[-1]))
    return out


@pytest.mark.parametrize("ros_type", sorted(JAZZY))
def test_the_type_is_in_the_bundle_under_its_dds_name(ros_type):
    from strands_robots.rtps.mangling import dds_type_name

    assert idl.get_type(ros_type).__idl_typename__ == dds_type_name(ros_type)


@pytest.mark.parametrize("ros_type", sorted(JAZZY))
def test_the_field_order_and_kinds_are_the_jazzy_msg(ros_type):
    assert _layout(idl.get_type(ros_type)) == _expected(ros_type)


@pytest.mark.parametrize("ros_type", sorted(JAZZY))
def test_a_default_sample_round_trips_through_the_encoder(ros_type):
    cls = idl.get_type(ros_type)
    sample = cls()
    assert cls.deserialize(sample.serialize()) == sample


def test_an_odometry_sample_keeps_every_value():
    Odometry = idl.get_type("nav_msgs/msg/Odometry")
    from strands_robots.rtps.participant import _build_sample

    sample = _build_sample(
        Odometry,
        {
            "header": {"frame_id": "odom"},
            "child_frame_id": "base_link",
            "pose": {"pose": {"position": {"x": 1.25}}, "covariance": [0.5] * 36},
            "twist": {"twist": {"angular": {"z": -0.75}}},
        },
    )
    back = Odometry.deserialize(sample.serialize())
    assert (back.child_frame_id, back.pose.pose.position.x, back.twist.twist.angular.z) == ("base_link", 1.25, -0.75)
    assert list(back.pose.covariance) == [0.5] * 36
