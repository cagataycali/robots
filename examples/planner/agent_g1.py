#!/usr/bin/env python3
"""Steer a Unitree G1 from natural language via a Strands agent.

An :class:`~strands_robots.planning.inputs.agent.AgentInput` gives the agent a
``set_locomotion_intent`` tool; the agent decomposes a goal string like
"walk forward, then switch to stealth and slow down" into a timed stream of
locomotion commands that the planner feeds to the WBC policy.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    MUJOCO_GL=egl python examples/planner/agent_g1.py "walk then crawl then stand" \
        --checkpoint /path/to/grootwbc-g1
"""

from __future__ import annotations

import argparse

from strands import Agent

from strands_robots import Robot
from strands_robots.planning import KinematicPlanner
from strands_robots.planning.inputs import AgentInput


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal", help="Natural-language locomotion goal")
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--mp4", default="/tmp/g1_agent.mp4")
    args = parser.parse_args()

    agent = Agent()
    planner = KinematicPlanner(AgentInput(agent, args.goal))
    robot = Robot("unitree_g1", mode="sim")
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
