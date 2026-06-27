#!/usr/bin/env python3
"""A Strands agent driving a real ROS 2 turtlesim through the use_ros tool.

This is the natural-language end of the showcase: instead of calling use_ros
directly, a Bedrock-backed agent is given the tool and a task in plain English,
and it autonomously sequences the publish/echo calls to draw a shape and report
the final pose.

Requires (inside a sourced ROS 2 env with turtlesim running):
    pip install strands-agents
    export AWS_BEARER_TOKEN_BEDROCK=...        # or any boto3 credential chain
    export STRANDS_MODEL_ID=global.anthropic.claude-opus-4-8   # optional
    export AWS_REGION=us-east-1

Run:
    python3 agent_drive.py
"""

from __future__ import annotations

import json
import os
import time

from strands import Agent
from strands.models import BedrockModel

from strands_robots.tools.use_ros import use_ros


def pose() -> tuple[float, float, float] | None:
    r = use_ros(action="echo", topic="/turtle1/pose", type="turtlesim/msg/Pose", count=1, timeout=3.0)
    if r["status"] != "success":
        return None
    sample = json.loads(r["content"][0]["text"].split(":\n", 1)[1])[0]
    return round(sample["x"], 2), round(sample["y"], 2), round(sample["theta"], 2)


def main() -> None:
    print("pose BEFORE:", pose())

    model = BedrockModel(
        model_id=os.getenv("STRANDS_MODEL_ID", "global.anthropic.claude-opus-4-8"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    agent = Agent(
        model=model,
        tools=[use_ros],
        system_prompt=(
            "You pilot a ROS 2 turtlesim turtle via the use_ros tool. Publish "
            "geometry_msgs/msg/Twist to /turtle1/cmd_vel: use_ros(action='publish', "
            "topic='/turtle1/cmd_vel', type='geometry_msgs/msg/Twist', "
            "fields={'linear':{'x':SPEED}, 'angular':{'z':TURN}}, count=N, rate=10.0). "
            "count/rate = seconds of motion; linear x in m/s, angular z in rad/s (left=+). "
            "Take real actions with the tool, one step at a time, with brief commentary."
        ),
    )

    task = (
        "Draw a square: 4 forward segments of ~2 seconds each (linear x=2.0), with a "
        "90-degree LEFT turn (angular z ~1.57, ~1 second) between each segment. After "
        "the 4th side, stop the turtle (publish zero velocity once), then echo the final pose."
    )
    print("\n=== AGENT DRIVING (square) ===")
    result = agent(task)
    print("\n=== FINAL ===\n" + str(result)[:800])
    time.sleep(1)
    print("\npose AFTER:", pose())


if __name__ == "__main__":
    main()
