#!/usr/bin/env python3
"""End-to-end VERA MimicGen rollout on a Panda arm in MuJoCo (records an MP4).

The most faithful VERA demonstration on a *real arm*: a Franka Emika **Panda**
(MimicGen's robot) in a strands-robots :class:`Simulation`, driven by the real
VERA MimicGen policy — a WAN video planner that dreams the next frames, an
AllTracker point tracker, and a Jacobian inverse-dynamics model that emits 6-DoF
**end-effector deltas** (+ gripper). strands-robots' VERA IK bridge converts each
delta into Panda joint targets (mink), auto-discovering the end-effector frame
from the compiled MuJoCo model — so no manual IK wiring is needed.

```
WAN dream  ──▶  AllTracker  ──▶  Jacobian IDM  ──▶  eef-delta chunk (7-D)
                                                          │  (VERA IK bridge)
                                                          ▼
                                              Panda joint targets ──▶ MuJoCo
```

Prerequisites
-------------
1. Host client + sim deps::

     uv pip install -e '.[sim-mujoco]' websockets msgpack

2. Checkpoints (VERA Wave-1 + the frozen WAN 2.1 base)::

     hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
     hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

3. The MimicGen server (holds the GPU; serves ws on :8800). It bundles
   AllTracker + the offline checkpoint resolver::

     docker build -f strands_robots/policies/vera/docker/Dockerfile -t strands-vera-server:latest .
     docker run --rm --gpus all --ipc=host -p 8800:8800 \
         -v "$PWD/vera-ckpts":/ckpts:ro -v "$PWD/Wan2.1-T2V-1.3B":/wan:ro \
         -e VERA_EMBODIMENT=mimicgen -e USE_OFFLINE_RESOLVE=1 \
         strands-vera-server:latest

Run
---
::

    MUJOCO_GL=egl python examples/vera_mimicgen_panda/rollout.py \
        --record examples/vera_mimicgen_panda/artifacts/mimicgen_panda.mp4

Verified on an L40S: the WAN planner + AllTracker + Jacobian IDM return a 7-D
eef-delta chunk, the VERA IK bridge solves it onto the Panda's 7 arm joints +
gripper, the arm moves under the policy, and an MP4 is recorded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_scene(robot: str = "panda", mesh: bool = False):
    """Panda arm + a 2-block stacking setup + MimicGen's two camera views.

    VERA MimicGen conditions on ``agentview_image`` (a fixed scene camera) and
    ``robot0_eye_in_hand_image`` (a wrist camera) — we name our cameras to match
    so the provider's view auto-resolution lines them up. The policy emits 6-DoF
    eef-deltas + gripper; the VERA IK bridge (auto-configured via the sim's
    ``bind_policy_sim_context`` hook) maps them onto the Panda's joints.
    """
    from strands_robots import Robot

    sim = Robot(robot, mesh=mesh)
    # MimicGen 2-block stacking props.
    sim.add_object(
        name="red_block",
        shape="box",
        position=[0.45, -0.05, 0.025],
        size=[0.025, 0.025, 0.025],
        color=[0.9, 0.2, 0.2, 1],
        mass=0.05,
    )
    sim.add_object(
        name="green_block",
        shape="box",
        position=[0.45, 0.10, 0.025],
        size=[0.03, 0.03, 0.02],
        color=[0.2, 0.8, 0.2, 1],
        mass=0.08,
    )
    # Seed a natural Panda "ready" pose. The default all-zeros config is a
    # near-singular straight-up pose that sits half out of frame — a Cosmos3
    # reasoner pass on an earlier rollout flagged the arm as "static" purely
    # because the (large) motion happened off-camera. A tabletop-ready pose
    # keeps the arm in view so the rollout reads as purposeful manipulation.
    ready = {
        "joint1": 0.0,
        "joint2": -0.4,
        "joint3": 0.0,
        "joint4": -2.0,
        "joint5": 0.0,
        "joint6": 1.6,
        "joint7": 0.8,
    }
    try:
        sim.send_action(ready, robot_name=robot, n_substeps=200)
    except Exception:
        pass  # best-effort; rollout still runs from the default pose
    # Camera names match MimicGen's view keys (scene + wrist), framed on the
    # tabletop workspace so the arm + both blocks are visible.
    sim.add_camera(name="agentview_image", position=[1.1, 0.0, 0.7], target=[0.4, 0.0, 0.15])
    sim.add_camera(name="robot0_eye_in_hand_image", position=[0.5, 0.0, 0.8], target=[0.45, 0.0, 0.05])
    return sim


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="VERA server host.")
    p.add_argument("--port", type=int, default=8800, help="VERA MimicGen ws port.")
    p.add_argument("--instruction", default="stack the red block on the green block")
    p.add_argument("--n-steps", type=int, default=40, help="Control steps to roll out.")
    p.add_argument("--control-frequency", type=float, default=20.0, help="MimicGen runs ~20 Hz.")
    p.add_argument("--action-horizon", type=int, default=10, help="MimicGen exec chunk = 10.")
    p.add_argument("--robot", default="panda", help="Arm asset (Panda == MimicGen robot).")
    p.add_argument("--ik-smoothing", type=float, default=0.4, help="EMA on IK joint targets (0=off; damps jitter).")
    p.add_argument(
        "--record",
        metavar="MP4",
        default="examples/vera_mimicgen_panda/artifacts/mimicgen_panda.mp4",
        help="Record the rollout to this MP4 (set '' to disable).",
    )
    p.add_argument(
        "--server-mode",
        choices=["subprocess", "docker", "attach"],
        default="attach",
        help="attach (default): connect to a running server; docker/subprocess: let the provider manage it.",
    )
    p.add_argument("--ckpt-root", default=None, help="VERA checkpoint root (docker/subprocess modes).")
    p.add_argument("--mesh", action="store_true", help="Higher-fidelity mesh rendering.")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        sim = _build_scene(robot=args.robot, mesh=args.mesh)
    except ImportError as e:
        print(f"Missing sim deps: {e}\nInstall: uv pip install -e '.[sim-mujoco]'", file=sys.stderr)
        return 2

    policy_config: dict = {
        "embodiment": "mimicgen",
        "host": args.host,
        "server_port": args.port,
        "ik_smoothing": args.ik_smoothing,
        # The sim's bind_policy_sim_context hook hands the MjModel + namespace to
        # the provider, which auto-discovers the Panda end-effector frame and
        # configures the IK bridge — eef-deltas become joint targets, no manual
        # action_mapping needed.
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
        video = {"path": str(out), "fps": 20, "camera": "agentview_image", "width": 512, "height": 512}
        print(f"Recording rollout -> {out}")

    print(f"Rolling out VERA MimicGen -> {args.robot} for {args.n_steps} steps (server={args.server_mode}) ...")
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
    for c in result.get("content", []):
        if c.get("text"):
            print(c["text"])
    if video and Path(video["path"]).exists():
        size = Path(video["path"]).stat().st_size
        print(f"✅ Rollout video: {video['path']} ({size / 1024:.0f} KB)")
    return 0 if result.get("status") in ("success", "completed", "ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
