#!/usr/bin/env python3
"""A Strands agent driving a real ROS 2 turtlesim through the use_ros tool.

The natural-language end of the showcase: a Bedrock-backed agent is given the
use_ros tool and asked, in plain English, to drive a clean SQUARE using
closed-loop control - it echoes the pose after every move, computes the heading
error, and issues corrective turns until each corner is within tolerance. All
ROS 2 I/O is in-process through rclpy (no ros2 CLI).

The script resets the canvas and teleports the turtle to a known corner first
(both via use_ros service_call - note teleport_absolute is resolved dynamically
and is not even in any static registry), then hands control to the agent.

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

from typing import Any, Callable

from strands_robots.tools.use_ros import use_ros as _use_ros_tool

# The @tool decorator returns a DecoratedFunctionTool instance whose
# __call__ static analysis (CodeQL) does not recognize. Bind it to a
# Callable-typed name so the call sites are seen as calling a function.
use_ros: Callable[..., dict[str, Any]] = _use_ros_tool


def pose() -> tuple[float, float, float] | None:
    r = use_ros(action="echo", topic="/turtle1/pose", type="turtlesim/msg/Pose", count=1, timeout=2.0)
    if r["status"] != "success":
        return None
    sample = json.loads(r["content"][0]["text"].split(":\n", 1)[1])[0]
    return round(sample["x"], 3), round(sample["y"], 3), round(sample["theta"], 3)


def main() -> None:
    # Clean canvas + known start corner facing east. teleport_absolute is resolved
    # dynamically by use_ros - it is not in any static type registry.
    use_ros(action="service_call", service="/reset", type="std_srvs/srv/Empty", fields={})
    time.sleep(0.5)
    use_ros(
        action="service_call",
        service="/turtle1/teleport_absolute",
        type="turtlesim/srv/TeleportAbsolute",
        fields={"x": 3.0, "y": 3.0, "theta": 0.0},
    )
    time.sleep(0.5)
    print("start pose:", pose())

    model = BedrockModel(
        model_id=os.getenv("STRANDS_MODEL_ID", "global.anthropic.claude-opus-4-8"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    agent = Agent(
        model=model,
        tools=[use_ros],
        system_prompt=(
            "You pilot a ROS 2 turtlesim turtle via the use_ros tool, using CLOSED-LOOP control. "
            "Drive: use_ros(action='publish', topic='/turtle1/cmd_vel', type='geometry_msgs/msg/Twist', "
            "fields={'linear':{'x':SPEED},'angular':{'z':TURN}}, count=N, rate=10.0). "
            "Sense: use_ros(action='echo', topic='/turtle1/pose', type='turtlesim/msg/Pose', count=1) "
            "returns x,y,theta (theta in radians, -pi..pi, 0=east, +=counterclockwise). "
            "ALWAYS echo the pose after each move and correct: if a turn overshoots or undershoots the "
            "target heading, issue a small corrective turn before driving forward. "
            "Forward speed x=1.5; for turns use small bursts (count=3-5 at z=1.5) then re-check heading. "
            "Be precise; verify each corner with an echo. Keep commentary to one short line per action."
        ),
    )
    task = (
        "Drive the turtle in a clean SQUARE with ~2.5 unit sides. Start heading east (theta=0). "
        "For each of 4 sides: drive forward ~2.5 units (check pose to confirm distance), then turn "
        "LEFT to the next cardinal heading (0 -> +1.571 -> +-3.142 -> -1.571 -> back to 0), using "
        "closed-loop echo+correct so each heading is within ~0.05 rad before driving the next side. "
        "After completing the square, stop (publish zero velocity) and echo the final pose."
    )
    print("\n=== AGENT DRIVING A CLOSED-LOOP SQUARE ===")
    result = agent(task)
    print("\n=== FINAL ===\n" + str(result)[:1000])
    time.sleep(1)
    print("\nfinal pose:", pose())


if __name__ == "__main__":
    main()
