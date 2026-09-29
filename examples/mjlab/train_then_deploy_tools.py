"""train_policy(rsl_rl) -> run_policy(rsl_rl_onnx): the agent's two tool calls, verbatim.

Calls the SAME ``@tool`` functions a Strands Agent calls, with the same
arguments, and prints their JSON results, so the transcript is reproducible
without an LLM in the loop. ``--agent`` runs the real Agent (Bedrock Claude)
over the two tools instead and prints its messages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from strands_robots import Robot
from strands_robots.tools.run_policy import run_policy
from strands_robots.tools.train_policy import train_policy


def _payload(res: dict) -> dict:
    for c in res.get("content", []):
        if "json" in c:
            return c["json"]
    return {}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Strands-Reach-SO101")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend", default="mjlab")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--json-out")
    a = p.parse_args(argv)

    t0 = time.monotonic()
    train_res = train_policy(
        action="train",
        provider="rsl_rl",
        output_dir=a.output_dir,
        steps=a.iterations,
        batch_size=a.num_envs,
        save_freq=a.iterations,
        seed=0,
        extra={"task": a.task},
    )
    print("== train_policy ==")
    print(json.dumps(train_res, indent=1, default=str)[:4000])
    tp = _payload(train_res)
    onnx = tp.get("exported_model")
    if train_res.get("status") != "success" or not onnx:
        sys.exit("train_policy did not produce an ONNX artifact")

    robot = Robot("so101", backend=a.backend, num_envs=1) if a.backend == "mjlab" else Robot("so101", backend=a.backend)
    try:
        run_res = run_policy(
            robot,
            policy_provider="rsl_rl_onnx",
            policy_config={"onnx_path": onnx, "robot": "so101", "target": [0.20, 0.05, 0.15]},
            n_episodes=a.episodes,
            n_steps=100,
            control_frequency=50.0,
            action_horizon=1,
            seed=0,
            dataset_root=f"{a.output_dir}/rollout_{a.backend}",
            dataset_repo_id="cagataydev/tmp-train-then-deploy",
            dataset_task="reach the target",
            dataset_fps=50,
        )
    finally:
        robot.cleanup()
    print("== run_policy ==")
    print(json.dumps(run_res, indent=1, default=str)[:4000])
    out = {
        "train": tp,
        "train_status": train_res.get("status"),
        "run": _payload(run_res),
        "run_status": run_res.get("status"),
        "wall_s": round(time.monotonic() - t0, 1),
    }
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
