#!/usr/bin/env python3
"""Steer a Unitree G1 from the keyboard in real time (WASD + style keys).

Bindings: w/s forward/back, a/d strafe, q/e turn, r/f height, space halt,
1-8 movement style, ESC stop. The keyboard reader runs off the control loop, so
input never stalls locomotion.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    MUJOCO_GL=egl python examples/planner/keyboard_g1.py --checkpoint /path/to/grootwbc-g1
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.planning import KinematicPlanner
from strands_robots.planning.inputs import KeyboardInput


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--mp4", default="/tmp/g1_keyboard.mp4")
    args = parser.parse_args()

    robot = Robot("unitree_g1", mode="sim")
    planner = KinematicPlanner(KeyboardInput())
    print("Steer the G1: WASD move, QE turn, RF height, 1-8 style, ESC stop.")
    result = robot.run_policy(
        robot_name="unitree_g1",
        policy_provider="wbc",
        policy_config={"checkpoint": args.checkpoint, "walk": True},
        planner=planner,
        duration=args.duration,
        control_frequency=50.0,
        video={"path": args.mp4, "fps": 30, "camera": "default", "width": 640, "height": 480},
    )
    print(result["content"][0]["text"])
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
