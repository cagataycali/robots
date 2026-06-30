#!/usr/bin/env python3
"""Steer a Unitree G1 through a scripted locomotion sequence and record it.

Self-contained: owns its own timed schedule of locomotion goals and drives the
WBC policy with short-horizon ``run_policy`` calls, one per segment. Each call
passes the goal through the well-known ``policy_kwargs`` channel
(``target_velocity`` / ``locomotion_style``) - the same channel every locomotion
provider reads. No library locomotion abstraction; the closed loop runs at this
script's cadence (re-issue ``run_policy`` to change the goal). Headless-friendly
(no TTY/agent) so it doubles as the reproducible demo artifact.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    # checkpoint dir with policy.onnx (+ walk_policy.onnx); see docs/policies/wbc.md
    MUJOCO_GL=egl python examples/locomotion/scripted_g1.py \
        --checkpoint /path/to/grootwbc-g1 --mp4 /tmp/g1_locomotion.mp4
"""

from __future__ import annotations

import argparse

from strands_robots import Robot

# A short steer: forward, veer left, forward faster, then halt. Each entry is
# (duration_s, policy_kwargs) - a plain goal dict forwarded verbatim to the
# policy. ``target_velocity`` is [vx, vy, wz] (planar velocity + yaw rate).
SCHEDULE: list[tuple[float, dict[str, list[float]]]] = [
    (2.0, {"target_velocity": [0.4, 0.0, 0.0]}),
    (2.0, {"target_velocity": [0.4, 0.0, 0.5]}),
    (2.0, {"target_velocity": [0.6, 0.0, 0.0]}),
    (2.0, {"target_velocity": [0.0, 0.0, 0.0]}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--mp4", default="/tmp/g1_locomotion.mp4")
    args = parser.parse_args()

    robot = Robot("unitree_g1", mode="sim")
    policy_config = {"checkpoint": args.checkpoint, "walk": True}
    status = 0
    for i, (duration, goal) in enumerate(SCHEDULE):
        # One short-horizon segment per goal; append each segment to one MP4.
        result = robot.run_policy(
            robot_name="unitree_g1",
            policy_provider="wbc",
            policy_config=policy_config,
            policy_kwargs=goal,
            duration=duration,
            control_frequency=50.0,
            video={"path": args.mp4, "fps": 30, "camera": "default", "width": 640, "height": 480},
        )
        print(f"segment {i} goal={goal}: {result['content'][0]['text']}")
        if result.get("status") != "success":
            status = 1
            break
    return status


if __name__ == "__main__":
    raise SystemExit(main())
