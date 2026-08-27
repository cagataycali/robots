#!/usr/bin/env python3
"""Evaluate a microduck_rl ONNX export in strands-robots - the RL -> eval loop.

`microduck_rl` trains a locomotion policy on mjlab (MuJoCo Warp + PPO) and runs
``scripts/export.py`` to emit a self-describing ONNX: the graph bakes in
observation normalization, and ``attach_metadata_to_onnx`` stamps ``joint_names``,
``default_joint_pos``, ``action_scale`` and ``command_names`` into the model's
metadata. :class:`~strands_robots.policies.microduck.MicroduckPolicy` reads
exactly those keys on first use, so a fresh export drops straight into the
standard ``run_policy`` seam with nothing to configure by hand.

This script is the eval half of that loop: point it at any ``export.py`` output
and it (1) prints what the policy self-configured to, (2) runs a MuJoCo rollout
with a body-tracking chase camera to MP4, and (3) optionally records the rollout
as a LeRobotDataset. Train anywhere (a GPU box or the repo's HF job), then::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python examples/microduck/eval_rl_policy.py --onnx output.onnx --record

Dependencies:
  pip install "strands-robots[sim-mujoco,microduck]"          # rollout + video
  pip install "strands-robots[sim-mujoco,microduck,lerobot]"  # + --record
"""

from __future__ import annotations

import argparse
import os

import onnxruntime as ort

from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy


def _report_metadata(onnx_path: str) -> None:
    """Print what the export self-describes - the contract MicroduckPolicy reads."""
    meta = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    ins = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).get_inputs()[0]
    print(f"  input {ins.name}{ins.shape}")
    for key in ("joint_names", "default_joint_pos", "action_scale", "command_names"):
        value = meta.get(key, "<missing>")
        print(f"  {key}: {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="a microduck_rl export.py output (.onnx)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--vx", type=float, default=0.25, help="forward velocity command, m/s")
    ap.add_argument("--video", default="/tmp/microduck_out/rl_eval.mp4")
    ap.add_argument("--record", action="store_true", help="also record a LeRobotDataset")
    ap.add_argument("--root", default="/tmp/microduck_rl_eval_ds")
    args = ap.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    if not os.path.exists(onnx_path):
        raise SystemExit(f"no such ONNX: {onnx_path}")

    print(f"== microduck_rl export -> MicroduckPolicy ==\n{onnx_path}")
    _report_metadata(onnx_path)
    # The policy configures itself from the metadata above - no manual wiring.
    policy = MicroduckPolicy(onnx_path=onnx_path)

    sim = Robot("microduck", mesh=False)
    sim.reset()
    sim.add_camera(
        name="chase", position=[0.0, -0.8, 0.4], target=[0, 0, 0.0],
        parent_body="microduck/trunk_base", fov=55,
    )

    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    if args.record:
        start = sim.start_recording(
            repo_id="local/microduck_rl_eval", root=args.root, fps=50,
            task="rl policy walk forward", overwrite=True, cameras=["chase"],
        )
        if start["status"] != "success":
            raise SystemExit(f"start_recording failed: {start['content'][0]['text']}")

    rollout = sim.run_policy(
        policy_object=policy,
        control_frequency=50.0, n_steps=args.steps,
        policy_kwargs={"target_velocity": [args.vx, 0.0, 0.0]},
        video=None if args.record else {"path": args.video, "camera": "chase", "fps": 50},
    )
    payload = rollout["content"][1]["json"] if len(rollout["content"]) > 1 else {}
    print(
        f"\nrollout: {rollout['status']} | steps={payload.get('steps_used')} "
        f"| action_errors={payload.get('action_errors')}"
    )

    if args.record:
        sim.save_episode()
        stop = sim.stop_recording()
        print(stop["content"][0]["text"])
    else:
        print(f"video: {payload.get('video_path', args.video)} ({payload.get('video_frames')} frames)")


if __name__ == "__main__":
    main()
