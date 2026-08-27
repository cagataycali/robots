#!/usr/bin/env python3
"""Let a Strands agent drive the Microduck and record a dataset, episode by episode.

This is the full agentic loop: a :class:`strands.Agent` is handed the
``Robot("microduck")`` simulation as a tool and told, in plain English, to
record a handful of walking demonstrations. The agent itself opens the
recording, runs Pollen's ``alpha_walking`` ONNX policy through the standard
``run_policy`` seam once per episode, saves each episode, and closes the
dataset - the human writes a sentence, the agent operates the robot.

The policy is driven by name (``policy_provider="microduck"``), so nothing but
a JSON tool call crosses the agent boundary - no Python objects.

Run::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib   # ffmpeg for video
    python examples/microduck/agent_record_episodes.py \
        --onnx "$PWD/../microduck/policies/alpha_walking.onnx" \
        --episodes 2 --root /tmp/microduck_agent_ds

Dependencies:
  pip install "strands-robots[sim-mujoco,lerobot,microduck]" strands-agents
  A strands-agents model provider (Bedrock by default).
"""

from __future__ import annotations

import argparse
import os

from strands import Agent

from strands_robots import Robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../microduck/policies/alpha_walking.onnx")
        ),
        help="Path to Pollen's alpha_walking.onnx",
    )
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=150, help="control steps per episode")
    ap.add_argument("--fps", type=int, default=50, help="recording fps (== control_frequency)")
    ap.add_argument("--root", default="/tmp/microduck_agent_ds")
    ap.add_argument("--repo-id", default="local/microduck_agent_walk")
    args = ap.parse_args()

    # Robot() is itself a Strands tool: hand it straight to the Agent.
    sim = Robot("microduck", mesh=False)
    agent = Agent(tools=[sim])

    prompt = f"""You are operating a Pollen Microduck biped in a MuJoCo simulation, using the
microduck_sim tool. Collect a walking dataset, one episode at a time. Do exactly this, in order:

1. Add a camera named "chase" attached to body "microduck/trunk_base",
   position [0.0, -0.8, 0.4], target [0, 0, 0.0], fov 55, so the recording sees the duck.
2. Start a recording: repo_id="{args.repo_id}", root="{args.root}", fps={args.fps},
   task="walk forward", overwrite=true, cameras=["chase"].
3. Repeat for {args.episodes} episodes. For EACH episode:
     a. Run a policy: policy_provider="microduck",
        policy_config={{"onnx_path": "{args.onnx}"}},
        control_frequency={args.fps}, n_steps={args.steps},
        policy_kwargs={{"target_velocity": [0.25, 0.0, 0.0]}}.
     b. Save the episode (save_episode).
     c. Reset the simulation so the next episode starts clean.
3. Stop the recording and report how many episodes and frames landed on disk.

The recording fps and the policy control_frequency MUST be equal ({args.fps}) or the rollout is
refused. Report the final on-disk path and frame count in your answer."""

    result = agent(prompt)
    print("\n===== AGENT RESULT =====")
    print(result)


if __name__ == "__main__":
    main()
