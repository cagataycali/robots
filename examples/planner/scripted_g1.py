#!/usr/bin/env python3
"""Steer a Unitree G1 with a scripted KinematicPlanner and record the rollout.

Drives the WBC locomotion policy through the intent layer: a
:class:`~strands_robots.planning.inputs.scripted.ScriptedInput` emits a timed
velocity/style sequence, the :class:`~strands_robots.planning.kinematic.KinematicPlanner`
turns it into per-tick locomotion goals, and ``run_policy(planner=...)`` feeds
them to the policy. Headless-friendly (no TTY/agent) so it doubles as the
reproducible demo artifact.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    # checkpoint dir with policy.onnx (+ walk_policy.onnx); see docs/policies/wbc.md
    MUJOCO_GL=egl python examples/planner/scripted_g1.py \
        --checkpoint /path/to/grootwbc-g1 --mp4 /tmp/g1_planner.mp4 --duration 8
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.planning import KinematicPlanner, PlannerUpdate
from strands_robots.planning.inputs import ScriptedInput


def build_planner() -> KinematicPlanner:
    """A short steer: forward, veer left, forward faster, then halt."""
    schedule = [
        (0.0, PlannerUpdate(root_vel=(0.4, 0.0, 0.0), style="run")),
        (2.0, PlannerUpdate(root_vel=(0.4, 0.0, 0.5))),
        (4.0, PlannerUpdate(root_vel=(0.6, 0.0, 0.0))),
        (6.0, PlannerUpdate(stop=True)),
    ]
    return KinematicPlanner(ScriptedInput(schedule), max_speed=1.0, max_omega=2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--mp4", default="/tmp/g1_planner.mp4")
    args = parser.parse_args()

    robot = Robot("unitree_g1", mode="sim")
    result = robot.run_policy(
        robot_name="unitree_g1",
        policy_provider="wbc",
        policy_config={"checkpoint": args.checkpoint, "walk": True},
        planner=build_planner(),
        duration=args.duration,
        control_frequency=50.0,
        video={"path": args.mp4, "fps": 30, "camera": "default", "width": 640, "height": 480},
    )
    print(result["content"][0]["text"])
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
