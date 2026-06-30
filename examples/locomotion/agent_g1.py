#!/usr/bin/env python3
"""Steer a Unitree G1 from natural language via a Strands agent.

The product pattern: the agent holds the robot (``Agent(tools=[robot])``) and
steers it by issuing short-horizon ``run_policy`` calls, choosing the locomotion
goal (``target_velocity`` / ``locomotion_style``) for each segment, observing the
result, then deciding the next goal. That is the closed loop - at the agent's
own cadence - with no library locomotion abstraction: the goal channel is just
``policy_kwargs`` and the drive primitive is the robot's own ``run_policy`` tool.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    MUJOCO_GL=egl python examples/locomotion/agent_g1.py "walk forward, then switch to stealth" \
        --checkpoint /path/to/grootwbc-g1
"""

from __future__ import annotations

import argparse

from strands import Agent

from strands_robots import Robot

_SYSTEM = """You steer a simulated Unitree G1 humanoid toward a locomotion goal.

Use the robot's run_policy tool, one short segment (duration ~2s) at a time:
  run_policy(policy_provider="wbc",
             policy_config={"checkpoint": <ckpt>, "walk": true},
             policy_kwargs={"target_velocity": [vx, vy, wz], "locomotion_style": <style>},
             duration=2.0, control_frequency=50.0)
target_velocity is [forward, lateral, yaw_rate] in m/s and rad/s. locomotion_style
is one of run/happy/stealth/injured/hand_crawling/elbow_crawling/boxing (optional).
Issue one call, read the result, then issue the next with an updated goal until
the user's goal is met. Halt with target_velocity [0, 0, 0]."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal", help="Natural-language locomotion goal")
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    args = parser.parse_args()

    robot = Robot("unitree_g1", mode="sim")
    agent = Agent(tools=[robot], system_prompt=_SYSTEM)
    agent(f"WBC checkpoint dir is {args.checkpoint!r}. Goal: {args.goal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
