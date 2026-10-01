"""An agent trains a reach policy in mjlab, then deploys it on the so101 - two tools, one prompt.

    python examples/mjlab/train_then_deploy_agent.py --output-dir runs/agent_reach

The agent gets ``train_policy`` (provider ``rsl_rl`` -> mjlab PPO -> ONNX) and a
``deploy_policy`` tool that is ``run_policy`` bound to ``Robot("so101",
backend="mjlab")``. Nothing in the prompt names an ONNX file: the agent has to
read ``exported_model`` out of the first tool's result and hand it to the
second, which is the loop the owner asked for (train -> deploy without a human
copying paths). The transcript (every tool call, its arguments and results) is
written to ``--transcript``. Needs Bedrock credentials (``AWS_BEARER_TOKEN_BEDROCK``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from strands import Agent
from strands.tools.decorator import tool

from strands_robots import Robot
from strands_robots.tools.run_policy import run_policy
from strands_robots.tools.train_policy import train_policy

PROMPT = """You have a MuJoCo-Warp simulator with an SO-101 arm.
1. Train a reach policy with train_policy: provider "rsl_rl", extra {{"task": "Strands-Reach-SO101"}},
   steps={its}, batch_size={envs}, save_freq={its}, seed=0, output_dir "{out}".
2. Read the exported_model path from the result and deploy it with deploy_policy for 3 episodes.
3. Report: training wall time, Metrics/reach/position_error, and the episodes/frames the deploy recorded
   (use the tool's numbers only)."""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--transcript")
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    robot = Robot("so101", backend="mjlab", num_envs=1)
    calls: list[dict[str, Any]] = []

    @tool
    def deploy_policy(onnx_path: str, n_episodes: int = 3, n_steps: int = 100) -> dict[str, Any]:
        """Roll out a trained rsl_rl ONNX actor on the so101 (mjlab backend) and record a LeRobot dataset.

        Args:
            onnx_path: The ``exported_model`` path returned by ``train_policy``.
            n_episodes: Episodes to record (each becomes one parquet episode).
            n_steps: Control ticks per episode at 50 Hz.
        """
        res = run_policy(
            robot,
            policy_provider="rsl_rl_onnx",
            policy_config={"onnx_path": onnx_path, "robot": "so101", "target": [0.20, 0.05, 0.15]},
            n_episodes=n_episodes,
            n_steps=n_steps,
            control_frequency=50.0,
            action_horizon=1,
            seed=0,
            dataset_root=os.path.join(a.output_dir, "rollout_mjlab"),
            dataset_repo_id="cagataydev/tmp-agent-train-then-deploy",
            dataset_task="reach the target",
            dataset_fps=50,
        )
        calls.append(
            {"tool": "deploy_policy", "args": {"onnx_path": onnx_path, "n_episodes": n_episodes}, "result": res}
        )
        return res

    agent = Agent(tools=[train_policy, deploy_policy])
    t0 = time.monotonic()
    try:
        result = agent(PROMPT.format(its=a.iterations, envs=a.num_envs, out=a.output_dir))
    finally:
        robot.cleanup()
    wall = time.monotonic() - t0

    # Recover the train_policy call(s) from the agent's own message history.
    tool_uses = []
    for m in agent.messages:
        for c in m.get("content", []):
            if "toolUse" in c:
                tool_uses.append({"name": c["toolUse"]["name"], "input": c["toolUse"]["input"]})
            if "toolResult" in c:
                tool_uses.append({"result_for": c["toolResult"]["toolUseId"], "content": c["toolResult"]["content"]})
    transcript = {
        "prompt": PROMPT.format(its=a.iterations, envs=a.num_envs, out=a.output_dir),
        "final_answer": str(result),
        "tool_uses": tool_uses,
        "deploy_calls": calls,
        "wall_s": round(wall, 1),
        "model": getattr(getattr(agent, "model", None), "config", None),
    }
    if a.transcript:
        with open(a.transcript, "w") as f:
            json.dump(transcript, f, indent=1, default=str)
    print(json.dumps({"wall_s": transcript["wall_s"], "n_tool_uses": len([t for t in tool_uses if "name" in t])}))
    print(str(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
