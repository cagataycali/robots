#!/usr/bin/env python3
"""Drive the MuJoCo SO-101 with FLUX 3 Action and record LeRobot v3 episodes.

End-to-end smoke test for the ``flux3_action`` policy provider: Black Forest
Labs' ``black-forest-labs/flux-3-action-so101`` checkpoint runs in-process
(no policy server) and drives ``Robot("so101", mode="sim")`` at 30 Hz through
the same ``run_policy`` path the agent tool uses. With ``--episodes N`` the
rollouts are recorded through ``start_recording`` (LeRobot v3: parquet + one
MP4 per camera) and optionally pushed to a PRIVATE Hugging Face dataset.

Prerequisites
-------------
1. Sim + recorder + the FLUX 3 Action inference library (git-only, hence the
   empty ``flux3`` extra) and NATTEN for your torch/CUDA build:

     pip install 'strands-robots[sim-mujoco,lerobot]'
     pip install 'flux-action[encoders] @ git+https://github.com/black-forest-labs/flux-action'
     pip install natten==0.21.6 -f https://whl.natten.org   # pick the wheel for your torch+cu

2. A CUDA GPU with ~22 GB free (3.9B transformer + video VAE + text encoder
   in bf16). Weights (~14 GB) download on first use.

Run
---
    MUJOCO_GL=egl python examples/vla/flux3_action_so101_sim.py --seconds 8
    MUJOCO_GL=egl python examples/vla/flux3_action_so101_sim.py --episodes 10 \
        --repo-id you/f3a-so101-mujoco --push

Verified on a Jetson AGX Thor (sm_110, torch 2.11+cu130): 450/450 actions per
15 s episode, tick p50 33 ms; each replan (every 32 ticks) costs ~5 s because
the published NATTEN wheels carry no sm_110 kernel and the provider falls back
to flex attention, so sim time runs ~8x slower than wall clock there.

Unit frame
----------
The checkpoint speaks lerobot SO-101 units (arm degrees, gripper percent) in
the frame of the community datasets it was tuned on. The MuJoCo ``so101``
model reports radians with a different zero (upper arm vertical, forearm
forward). ``Flux3ActionPolicy`` converts both ways through ``UnitAdapter``
(``joint_signs`` / ``joint_offsets_deg`` / ``gripper_range``; defaults are the
MuJoCo ``so101`` calibration) and warns once when the converted state leaves
the checkpoint's q01..q99 window. Episodes start from
``SO101_SIM_REST_QPOS_RAD``, the folded rest posture the training episodes
start from.

Agent
-----
The same rollout through one tool::

    from strands import Agent
    from strands_robots import Robot

    agent = Agent(tools=[Robot("so101", mode="sim")])
    agent(
        "Add cameras 'scene' and 'wrist' (wrist on body so101/gripper), a red "
        "cube at [0, -0.2, 0.02], then run the flux3_action policy for 8 seconds "
        "at 30 Hz with instruction 'pick up the red cube'."
    )
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

FPS = 30
TASK = "pick up the red cube"


def build_scene(robot: Any) -> None:
    """Two cameras in the roles the checkpoint expects and a cube in reach."""
    for step in (
        robot.add_camera(name="scene", position=[0.30, -0.62, 0.32], target=[0.0, -0.15, 0.06], width=256, height=256),
        # Gripper local frame: the jaws extend along -z, local +x is "up" at qpos 0.
        robot.add_camera(
            name="wrist",
            parent_body="so101/gripper",
            position=[0.06, 0.0, 0.0],
            target=[-0.008, 0.0, -0.16],
            width=256,
            height=256,
        ),
        robot.add_object(
            name="cube",
            shape="box",
            position=[0.0, -0.20, 0.02],
            size=[0.02, 0.02, 0.02],
            color=[0.9, 0.1, 0.1, 1],
            mass=0.03,
        ),
    ):
        if step["status"] != "success":
            raise SystemExit(f"scene refused: {step['content']}")


def park_at_rest(robot: Any) -> None:
    """Make the SO-101 rest posture the pose every ``reset()`` restores."""
    from strands_robots.policies.flux3_action.units import SO101_SIM_REST_QPOS_RAD

    record = robot._world.robots["so101"]
    record.home_qpos = {f"so101/{i + 1}": [q] for i, q in enumerate(SO101_SIM_REST_QPOS_RAD)}
    record.home_actuators = {f"so101/{i + 1}": (q, []) for i, q in enumerate(SO101_SIM_REST_QPOS_RAD)}
    robot.reset()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruction", default=TASK)
    parser.add_argument("--seconds", type=float, default=8.0, help="Sim seconds per episode.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes; >1 implies recording.")
    parser.add_argument("--mode", choices=("queued", "chunk"), default="queued")
    parser.add_argument("--repo-id", default="local/f3a-so101-mujoco", help="LeRobot dataset id when recording.")
    parser.add_argument("--root", default=None, help="On-disk dataset directory (default: $HF_LEROBOT_HOME/<repo-id>).")
    parser.add_argument("--record", action="store_true", help="Record even a single episode.")
    parser.add_argument("--push", action="store_true", help="Push the finished dataset to the Hub as PRIVATE.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    try:
        from strands_robots import Robot
    except ImportError as e:
        print(f"Missing deps: {e}\nInstall: pip install 'strands-robots[sim-mujoco,lerobot]'")
        return 2

    recording = args.record or args.episodes > 1
    robot = Robot("so101", mode="sim")
    try:
        build_scene(robot)
        park_at_rest(robot)
        if recording:
            started = robot.start_recording(
                repo_id=args.repo_id,
                task=args.instruction,
                fps=FPS,
                root=args.root,
                overwrite=True,
                cameras=["scene", "wrist"],
            )
            if started["status"] != "success":
                raise SystemExit(f"start_recording refused: {started['content']}")
        print(f"FLUX 3 Action -> so101 (MuJoCo), {args.episodes} x {args.seconds:g} s at {FPS} Hz, mode={args.mode}")
        ran = robot.run_policy(
            robot_name="so101",
            policy_provider="flux3_action",
            policy_config={"device": args.device, "mode": args.mode},
            instruction=args.instruction,
            duration=args.seconds,
            control_frequency=FPS,
            n_episodes=args.episodes,
            reset_between=True,
        )
        report = next((c["json"] for c in ran["content"] if "json" in c), {})
        print(
            f"status={ran['status']} actions_applied={report.get('actions_applied')} "
            f"action_errors={report.get('action_errors')} avg_inference_ms={report.get('avg_inference_ms')} "
            f"episodes_saved={report.get('episodes_saved')}"
        )
        if recording:
            stopped = robot.stop_recording()
            print("stop_recording:", stopped["status"], *(c.get("text", "") for c in stopped["content"]))
            if args.push and stopped["status"] == "success":
                # stop_recording finalizes and releases the recorder; publish the
                # finished on-disk dataset itself so the repo can be PRIVATE.
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                saved = robot._world._backend_state["last_save"]
                dataset = LeRobotDataset(saved["repo_id"], root=saved["root"])
                dataset.push_to_hub(private=True, tags=["strands-robots", "sim", "flux3-action", "so101"])
                print(f"pushed PRIVATE https://huggingface.co/datasets/{saved['repo_id']}")
    finally:
        robot.destroy()
    return 0 if ran["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
