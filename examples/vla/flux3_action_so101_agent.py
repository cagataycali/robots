#!/usr/bin/env python3
"""One tool, one prompt: a Strands Agent drives the SO-101 with FLUX 3 Action.

Goal: Show the single-tool agent flow for the ``flux3_action`` provider. The
whole robot is ONE Strands tool (``Robot(...)``); the agent builds the scene
and runs the policy from a text prompt. The sim and the real arm take the
same prompt because both tools accept ``policy_config`` on their policy
action (``run_policy`` in sim, ``execute`` / ``start`` on hardware).

Prerequisites:
  pip install 'strands-robots[sim-mujoco,lerobot]'
  pip install 'flux-action[encoders] @ git+https://github.com/black-forest-labs/flux-action'
  pip install natten==0.21.6 -f https://whl.natten.org   # wheel for your torch+cu
  A CUDA GPU with ~22 GB free; AWS credentials for Bedrock (or any
  strands-agents model provider).

Run (sim, MuJoCo, default):
  MUJOCO_GL=egl python examples/vla/flux3_action_so101_agent.py

Run (real SO-101, lerobot driver; the HITL gate asks before the arm moves):
  python examples/vla/flux3_action_so101_agent.py --real /dev/ttyACM0

Expected output: the agent's answer quoting the policy result
(``actions_applied`` at 30 Hz) and whether the arm moved.

Verified in sim on a Jetson AGX Thor (torch 2.11+cu130): 120/120 actions in
4 sim seconds, 4 tool calls, one agent turn. The ``--real`` path was not run
in that session; it rests on the hardware tool forwarding ``policy_config``
to the provider, which this example exercises the same way as the sim tool.
"""

from __future__ import annotations

import argparse
import os
import sys

from strands import Agent

from strands_robots import Robot

PROMPT_SIM = (
    "Add a camera named 'scene' at position [0.30, -0.62, 0.32] looking at "
    "[0.0, -0.15, 0.06], 256x256, and a camera named 'wrist' on parent body "
    "'so101/gripper' at [0.06, 0, 0] looking at [-0.008, 0, -0.16], 256x256. "
    "Add a red box named 'cube' at [0, -0.2, 0.02] with size [0.02, 0.02, 0.02]. "
    "Then run the policy provider 'flux3_action' with policy_config "
    '{"device": "cuda", "mode": "queued"} and instruction "pick up the red cube" '
    "for 4 seconds at 30 Hz on robot so101. Quote the policy result, then say in "
    "one sentence whether the arm moved and how many actions were applied."
)

# A lerobot-calibrated SO-101 reports degrees with no sign flips or offsets and a
# 0..100 gripper; the constructor defaults are the MuJoCo frame (radians, the
# sim's signs and offsets), so the real arm must name its frame or the policy
# reads degrees as radians.
PROMPT_REAL = (
    "Execute the policy provider 'flux3_action' with policy_config "
    '{"device": "cuda", "mode": "queued", "joint_units": "deg", '
    '"joint_signs": [1, 1, 1, 1, 1], "joint_offsets_deg": [0, 0, 0, 0, 0], '
    '"gripper_range": [0, 100]} and instruction "pick up the red cube" '
    "for 4 seconds at 30 Hz. Quote the result, then say whether the arm moved."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", metavar="PORT", help="Serial port of a calibrated SO-101; omit for MuJoCo.")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    if args.real:
        # Same provider, same prompt shape; the hardware tool's execute action
        # forwards policy_config to create_policy exactly as the sim tool does.
        cam = {"type": "opencv", "width": 640, "height": 480, "fps": 30}
        robot = Robot(
            "so101",
            mode="real",
            port=args.real,
            cameras={"scene": {**cam, "index_or_path": 0}, "wrist": {**cam, "index_or_path": 1}},
        )
        prompt = PROMPT_REAL
    else:
        robot = Robot("so101", mode="sim", mesh=False)
        prompt = PROMPT_SIM

    agent = Agent(tools=[robot])
    try:
        result = agent(prompt)
    finally:
        # Sim worlds are destroyed; the hardware tool releases its bus in cleanup().
        (robot.destroy if hasattr(robot, "destroy") else robot.cleanup)()
    print(f"Agent completed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
