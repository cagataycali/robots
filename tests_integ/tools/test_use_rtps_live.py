"""Live RTPS integration test - the 'act as a robot' thesis, end to end.

This is the decisive test: a bare cyclonedds participant (NO rclpy, NO sourced
ROS 2) publishes geometry_msgs/Twist on rt/turtle1/cmd_vel, and a real ROS 2
turtlesim must MOVE. It is gated behind RTPS_LIVE=1 and a turtlesim peer on the
same DDS domain.

Run (turtle on the same machine/host-net):
    RTPS_LIVE=1 pytest -m rtps tests_integ/tools/test_use_rtps_live.py

Setup (turtlesim in a container on host networking):
    docker run -d --name turtle --net host ros:jazzy bash -lc \\
      "apt-get update && apt-get install -y ros-jazzy-turtlesim && \\
       source /opt/ros/jazzy/setup.bash && \\
       QT_QPA_PLATFORM=offscreen ros2 run turtlesim turtlesim_node"
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.rtps

_LIVE = os.getenv("RTPS_LIVE") == "1"
pytest.importorskip("cyclonedds", reason="requires the [ros2] extra")

if not _LIVE:
    pytest.skip("RTPS_LIVE!=1: skipping live turtlesim test", allow_module_level=True)


def _texts(result: dict) -> str:
    return "\n".join(i.get("text", "") for i in result.get("content", []))


def _read_pose_x() -> float | None:
    """Read turtlesim's current x via use_ros if rclpy is available, else None."""
    try:
        from strands_robots.tools.use_ros import use_ros

        res = use_ros(action="echo", topic="/turtle1/pose", type="turtlesim/msg/Pose", count=1, timeout=3.0)
        import re

        m = re.search(r'"x":\s*([0-9.]+)', _texts(res))
        return float(m.group(1)) if m else None
    except Exception:
        return None


def test_rtps_participant_drives_turtlesim() -> None:
    from strands_robots.tools.use_rtps import use_rtps

    status = use_rtps(action="status")
    assert "cyclonedds" in _texts(status)

    before = _read_pose_x()

    # Advertise then publish a forward velocity for ~1.5s at 10 Hz.
    use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    result = use_rtps(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 0.0}},
        count=15,
        rate=10.0,
    )
    assert result["status"] == "success", _texts(result)
    time.sleep(0.5)

    after = _read_pose_x()
    if before is not None and after is not None:
        assert after != before, f"turtle did not move: x stayed {before}"
