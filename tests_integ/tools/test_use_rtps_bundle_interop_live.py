"""Every RTPS bundle type a mobile base exchanges, on the wire against a real ROS 2 node, both directions.

A bare cyclonedds participant (no rclpy) publishes each type and ``ros2 topic
echo`` in a ROS 2 container must print the sample; ``ros2 topic pub`` in the
container publishes it back and ``use_rtps echo`` must decode it. A field
layout that differs from the upstream ``.msg`` passes every unit test and fails
here, because a real node drops a sample it cannot decode.

Run (a ROS 2 Jazzy container on host networking):
    docker run -d --rm --name ros2-peer --net host --ipc host ros:jazzy-ros-base sleep infinity
    RTPS_LIVE=1 RTPS_ROS_CONTAINER=ros2-peer pytest -m rtps tests_integ/tools/test_use_rtps_bundle_interop_live.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.rtps

pytest.importorskip("cyclonedds", reason="requires the [ros2] extra")
if os.getenv("RTPS_LIVE") != "1" or not os.getenv("RTPS_ROS_CONTAINER"):
    pytest.skip("RTPS_LIVE!=1 or no RTPS_ROS_CONTAINER: skipping the live ROS 2 interop test", allow_module_level=True)

CONTAINER = os.environ["RTPS_ROS_CONTAINER"]

#: type -> (fields use_rtps publishes, a value ros2 must print), (YAML ros2 publishes, a key/value use_rtps must decode)
CASES = {
    "std_msgs/msg/String": ({"data": "hello ros"}, "data: hello ros", "{data: from_ros}", ("data", "from_ros")),
    "std_msgs/msg/Bool": ({"data": True}, "data: true", "{data: true}", ("data", True)),
    "std_msgs/msg/Int32": ({"data": -123456}, "data: -123456", "{data: 424242}", ("data", 424242)),
    "std_msgs/msg/Float32": ({"data": 1.5}, "data: 1.5", "{data: 2.25}", ("data", 2.25)),
    "std_msgs/msg/Float64": ({"data": -0.125}, "data: -0.125", "{data: -2.5}", ("data", -2.5)),
    "geometry_msgs/msg/PoseStamped": (
        {"header": {"frame_id": "map"}, "pose": {"position": {"x": 1.75}}},
        "x: 1.75",
        "{header: {frame_id: odom}}",
        ("header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "odom"}),
    ),
    "geometry_msgs/msg/TwistStamped": (
        {"twist": {"linear": {"x": 0.375}}},
        "x: 0.375",
        "{header: {frame_id: base_link}}",
        ("header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "base_link"}),
    ),
    "geometry_msgs/msg/PoseWithCovariance": (
        {"covariance": [0.0] * 35 + [6.5]},
        "- 6.5",
        "{pose: {position: {x: 9.0}}}",
        ("covariance", [0.0] * 36),
    ),
    "geometry_msgs/msg/TwistWithCovariance": (
        {"covariance": [0.0] * 35 + [7.5]},
        "- 7.5",
        "{twist: {linear: {x: 1.25}}}",
        ("covariance", [0.0] * 36),
    ),
    "nav_msgs/msg/Odometry": (
        {"child_frame_id": "base_link", "pose": {"pose": {"position": {"x": 1.25}}}},
        "child_frame_id: base_link",
        "{child_frame_id: base_footprint}",
        ("child_frame_id", "base_footprint"),
    ),
    "sensor_msgs/msg/LaserScan": (
        {"range_max": 10.0, "ranges": [1.0, 2.0, 3.5]},
        "- 3.5",
        "{range_max: 12.0, ranges: [1.5, 2.5]}",
        ("ranges", [1.5, 2.5]),
    ),
    "sensor_msgs/msg/Imu": (
        {"linear_acceleration": {"z": 9.75}},
        "z: 9.75",
        "{angular_velocity_covariance: [1,2,3,4,5,6,7,8,9]}",
        ("angular_velocity_covariance", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    ),
}


def _ros(cmd: str, timeout: int = 40) -> subprocess.Popen:
    return subprocess.Popen(
        ["docker", "exec", CONTAINER, "bash", "-lc", f"source /opt/ros/*/setup.bash && timeout {timeout} {cmd}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.mark.parametrize("ros_type", sorted(CASES))
def test_a_real_ros2_node_reads_what_use_rtps_publishes(ros_type):
    from strands_robots.tools.use_rtps import use_rtps

    fields, printed, _, _ = CASES[ros_type]
    topic = "/strands_interop/" + ros_type.split("/")[-1].lower()
    echo = _ros(f"ros2 topic echo --once {topic} {ros_type}")
    time.sleep(4)  # ros2 CLI startup + discovery
    result = use_rtps(action="publish", topic=topic, type=ros_type, fields=fields, count=30, rate=5)
    out = echo.communicate(timeout=60)[0]
    assert result["status"] == "success", result
    assert printed in out, out


@pytest.mark.parametrize("ros_type", sorted(CASES))
def test_use_rtps_decodes_what_a_real_ros2_node_publishes(ros_type):
    from strands_robots.tools.use_rtps import use_rtps

    _, _, yaml_in, (key, value) = CASES[ros_type]
    topic = "/strands_interop/" + ros_type.split("/")[-1].lower() + "_back"
    pub = _ros(f'ros2 topic pub -r 5 -t 40 {topic} {ros_type} "{yaml_in}"')
    try:
        result = use_rtps(action="echo", topic=topic, type=ros_type, count=1, timeout=15)
    finally:
        pub.kill()
    text = result["content"][0]["text"]
    samples = json.loads(text.split(":\n", 1)[1])
    assert samples, text
    assert samples[0][key] == value
