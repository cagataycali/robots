#!/usr/bin/env python3
"""Cross-process proof: drive a real ROS 2 turtlesim through use_rtps.

Runs INSIDE the proof container (see Dockerfile / docker-compose.yml). It:

1. reads turtlesim's pose before (via the ros2 CLI, the ground-truth observer),
2. drives /turtle1/cmd_vel using ``RtpsRobot`` -> ``use_rtps`` -> bare cyclonedds
   (NO rclpy on the publish side - only our ROS<->DDS mangling + IDL bundle),
3. reads the pose after and asserts the turtle moved.

A non-zero exit means the turtle did not move - the wire format is wrong.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

from strands_robots.mesh import RtpsRobot


def _ros_pose() -> tuple[float, float, float] | None:
    """Read turtlesim's (x, y, theta) via the ros2 CLI - the ground truth."""
    try:
        out = subprocess.run(
            ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && " + "timeout 5 ros2 topic echo --once /turtle1/pose"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except subprocess.TimeoutExpired:
        return None

    def g(k: str) -> str | None:
        m = re.search(rf"{k}:\s*([-0-9.]+)", out)
        return m.group(1) if m else None

    x, y, th = g("x"), g("y"), g("theta")
    if None in (x, y, th):
        return None
    return float(x), float(y), float(th)


def main() -> int:
    print("== reading pose BEFORE ==")
    before = _ros_pose()
    print(f"   before: {before}")
    if before is None:
        print("ERROR: could not read turtlesim pose - is the node running?", file=sys.stderr)
        return 2

    print("== driving via RtpsRobot -> use_rtps -> bare cyclonedds (no rclpy) ==")
    turtle = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    turtle.advertise()
    time.sleep(1.0)  # DDS discovery settle
    result = turtle.drive(linear=2.0, angular=2.0, duration=3.0)
    print(f"   drive result: {result['status']} - {result['content'][0]['text']}")
    turtle.stop()
    time.sleep(0.5)

    print("== reading pose AFTER ==")
    after = _ros_pose()
    print(f"   after:  {after}")
    if after is None:
        print("ERROR: could not read pose after driving", file=sys.stderr)
        return 2

    moved = abs(after[0] - before[0]) + abs(after[1] - before[1]) + abs(after[2] - before[2])
    print(f"== total pose delta: {moved:.4f} ==")
    if moved < 0.1:
        print("FAIL: turtle did not move - the RTPS wire format is wrong", file=sys.stderr)
        return 1

    print("\nPASS: a real ROS 2 turtlesim was driven by use_rtps over pure RTPS.")
    print(f"  x:     {before[0]:.3f} -> {after[0]:.3f}")
    print(f"  y:     {before[1]:.3f} -> {after[1]:.3f}")
    print(f"  theta: {before[2]:.3f} -> {after[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
