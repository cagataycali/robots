#!/usr/bin/env python3
"""SO101 pick-and-place driven by MolmoAct2 — strands_robots API only.

Demonstrates a complete VLA-driven manipulation loop using ONLY
``strands_robots`` imports. The Robot() factory handles lerobot robot
construction internally; create_policy() handles model loading, embodiment
mapping, and processor pipeline construction.

This example works identically with ``--mode sim`` and ``--mode real`` —
the same policy, the same observation loop, the same action dispatch. That
is the core value proposition of the strands_robots abstraction.

Hardware requirements (mode=real):
  - SO101 follower arm on a serial port
  - Front camera (OpenCV-compatible, index 0)
  - CUDA GPU for inference (or cpu with --device cpu)

Usage:
  export STRANDS_TRUST_REMOTE_CODE=1
  python molmoact2_so101_pickplace.py --task "Pick up the pen"
  python molmoact2_so101_pickplace.py --mode sim --device cpu --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("molmoact2_so101")

REPO = "allenai/MolmoAct2-SO100_101"


def main():
    ap = argparse.ArgumentParser(description="SO101 MolmoAct2 pick-and-place")
    ap.add_argument("--mode", choices=["real", "sim"], default="real",
                    help="Robot mode: 'real' for hardware, 'sim' for MuJoCo (default: real)")
    ap.add_argument("--port", default="/dev/ttyACM1",
                    help="Serial port for real hardware (ignored in sim mode)")
    ap.add_argument("--id", default=None,
                    help="Calibration ID / namespace (default: auto)")
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera index for observation (default: 0)")
    ap.add_argument("--task", default="Pick up the pen and place it on the paper")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--device", default="cuda", help="Inference device (default: cuda)")
    ap.add_argument("--dry-run", action="store_true", help="Run inference without sending actions")
    args = ap.parse_args()

    from strands_robots import Robot, create_policy

    # Construct the robot via the unified factory.
    # In real mode: wraps lerobot SO101Follower with camera config.
    # In sim mode: creates a MuJoCo simulation with the SO101 model.
    if args.mode == "real":
        robot = Robot(
            "so101",
            mode="real",
            port=args.port,
            **({"id": args.id} if args.id else {}),
            cameras={"front": {
                "type": "opencv",
                "index_or_path": args.camera,
                "width": 640,
                "height": 480,
                "fps": 30,
            }},
        )
    else:
        robot = Robot("so101", mode="sim")

    # Connect the hardware robot (sim is ready immediately).
    if args.mode == "real":
        log.info("Connecting SO101 @ %s ...", args.port)
        robot.robot.connect(calibrate=False)
        log.info("Connected. obs keys: %s", list(robot.robot.get_observation().keys()))

    # ONE call creates and configures the entire policy:
    #   - Detects MolmoAct2 transformers-native checkpoint
    #   - Loads 'so_real' embodiment (motor keys + camera renames)
    #   - Builds MolmoAct2Config, norm_tag, processor bridge
    #   - robot_state_keys auto-set from embodiment.action_keys
    policy = create_policy(REPO, embodiment="so_real", device=args.device)
    policy.reset()

    async def run():
        period = 1.0 / args.hz
        for step in range(args.steps):
            obs = robot.robot.get_observation()
            t = time.time()
            actions = await policy.get_actions(obs, args.task)
            dt = time.time() - t
            a = actions[0]
            log.info("step %d infer=%.2fs action=%s", step, dt,
                     {k: round(v, 1) for k, v in a.items()})
            if not args.dry_run:
                robot.robot.send_action(a)
            await asyncio.sleep(max(0, period - dt))

    try:
        asyncio.run(run())
    finally:
        if args.mode == "real":
            try:
                robot.robot.disconnect()
            except Exception as e:
                log.warning("disconnect: %s", str(e)[:80])
        log.info("Done.")


if __name__ == "__main__":
    main()
