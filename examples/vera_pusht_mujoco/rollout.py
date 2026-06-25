#!/usr/bin/env python3
"""End-to-end VERA PushT rollout in MuJoCo (records an MP4 artifact).

This is the headline smoke test for the ``vera`` policy provider: a planar
pusher in a strands-robots :class:`Simulation` is driven by the **real** VERA
PushT policy — a DFoT video planner that "dreams" the next frames plus a
Jacobian inverse-dynamics model that turns the dream into 2-D push velocities —
served by ``vera.server.start_vera_server`` over the WebSocket protocol.

The provider (host side, numpy>=2) needs **no** VERA install: the GPU stack
(torch 2.6 / CUDA, VGGT, DFoT) lives in the ``strands-vera-server`` container.

Prerequisites
-------------
1. Host client deps (sim + the provider's tiny ws client)::

     uv pip install -e '.[sim-mujoco]'
     uv pip install websockets msgpack            # the vera client transport

2. The VERA PushT server (holds the GPU). Either let the provider launch the
   container for you (``server_mode="docker"``), or start it yourself::

     # download the Wave-1 checkpoints once (~4 GB of the repo is PushT):
     hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts

     # build + run the server container (PushT serves ws on :8820):
     docker build -f strands_robots/policies/vera/docker/Dockerfile \
         -t strands-vera-server:latest .
     docker run --rm --gpus all --ipc=host -p 8820:8820 \
         -v "$PWD/vera-ckpts":/ckpts:ro -e VERA_EMBODIMENT=pusht \
         strands-vera-server:latest

Run
---
::

    # headless servers need an EGL/OSMesa GL backend for offscreen rendering
    MUJOCO_GL=egl python examples/vera_pusht_mujoco/rollout.py
    MUJOCO_GL=egl python examples/vera_pusht_mujoco/rollout.py \
        --record examples/vera_pusht_mujoco/artifacts/pusht_rollout.mp4

    # provider-managed container (no manual `docker run`):
    python examples/vera_pusht_mujoco/rollout.py \
        --server-mode docker --ckpt-root "$PWD/vera-ckpts"

Verified on a single L40S: the DFoT planner + Jacobian IDM return a 2-D push
velocity chunk (~2 s warm/chunk), the pusher moves under the policy, and an MP4
is recorded to ``artifacts/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_scene(robot: str = "dynamixel_2r", mesh: bool = False):
    """Scene for the PushT rollout.

    PushT is a *task-space* planar push: VERA's PushT policy emits a 2-D action
    chunk (``action_space="velocity"``, ``action_dim=2``, no gripper). Our sim's
    ``run_policy`` drives a registered robot's actuators, so we use a small
    **2-DoF** robot (``dynamixel_2r``: joints ``R1``/``R2``) as the pusher proxy
    — VERA's two action columns map onto its two joints. A top-down camera named
    ``image`` matches PushT's single view key.

    (A fully task-faithful PushT env — free-body puck + T block + goal-IoU
    reward — lives in VERA's own ``pusht_runner``; here we exercise the
    **provider → server → action** path through strands-robots' ``Simulation``.)
    """
    from strands_robots import Robot

    sim = Robot(robot, mesh=mesh)
    # Top-down camera named "image" — matches VERA PushT's single view key.
    sim.add_camera(name="image", position=[0.0, 0.0, 0.6], target=[0.0, 0.0, 0.0])
    return sim


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1", help="VERA server host.")
    p.add_argument("--port", type=int, default=8820, help="VERA PushT ws port.")
    p.add_argument("--instruction", default="push the T to the goal")
    p.add_argument("--n-steps", type=int, default=80, help="Control steps to roll out.")
    p.add_argument("--control-frequency", type=float, default=10.0, help="PushT runs ~10 Hz.")
    p.add_argument("--action-horizon", type=int, default=3, help="PushT exec chunk = 3.")
    p.add_argument(
        "--record",
        metavar="MP4",
        default="examples/vera_pusht_mujoco/artifacts/pusht_rollout.mp4",
        help="Record the rollout to this MP4 (set '' to disable).",
    )
    p.add_argument(
        "--server-mode", choices=["subprocess", "docker", "attach"], default="attach",
        help="attach: connect to an already-running server (default); "
        "docker: provider launches the strands-vera-server container; "
        "subprocess: provider launches a local `python -m vera.server...`.",
    )
    p.add_argument("--ckpt-root", default=None, help="VERA checkpoint root (docker/subprocess modes).")
    p.add_argument("--robot", default="dynamixel_2r", help="2-DoF pusher-proxy robot (PushT 2D action -> 2 joints).")
    p.add_argument("--mesh", action="store_true", help="Higher-fidelity mesh rendering.")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")  # headless-friendly default

    try:
        sim = _build_scene(robot=args.robot, mesh=args.mesh)
    except ImportError as e:
        print(f"Missing sim deps: {e}\nInstall: uv pip install -e '.[sim-mujoco]'", file=sys.stderr)
        return 2

    # Provider config: "attach" == auto_launch_server=False (talk to a running
    # server). docker/subprocess let the provider manage the server lifecycle.
    policy_config: dict = {
        "embodiment": "pusht",
        "host": args.host,
        "server_port": args.port,
        "image_keys": ["image"],  # our top-down camera maps to PushT's single view
    }
    if args.server_mode == "attach":
        policy_config["auto_launch_server"] = False
    else:
        policy_config["server_mode"] = args.server_mode
        policy_config["auto_launch_server"] = True
        if args.ckpt_root:
            policy_config["ckpt_root"] = args.ckpt_root

    video = None
    if args.record:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        video = {"path": str(out), "fps": 10, "camera": "image", "width": 512, "height": 512}
        print(f"Recording rollout -> {out}")

    print(f"Rolling out VERA PushT for {args.n_steps} steps (server={args.server_mode}) ...")
    result = sim.run_policy(
        robot_name=args.robot,
        policy_provider="vera",
        policy_config=policy_config,
        instruction=args.instruction,
        n_steps=args.n_steps,
        control_frequency=args.control_frequency,
        action_horizon=args.action_horizon,
        video=video,
    )
    print(f"Status: {result.get('status')}")
    if video and Path(video["path"]).exists():
        size = Path(video["path"]).stat().st_size
        print(f"✅ Rollout video: {video['path']} ({size / 1024:.0f} KB)")
    return 0 if result.get("status") in ("success", "completed", "ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
