#!/usr/bin/env python3
"""Steer a Unitree G1 through a scripted locomotion sequence and record it.

Self-contained: owns its own timed schedule of locomotion goals and drives the
WBC policy with short-horizon ``run_policy`` calls, one per segment. Each call
passes the goal through the well-known ``policy_kwargs`` channel
(``target_velocity`` / ``locomotion_style``) - the same channel every locomotion
provider reads. No library locomotion abstraction; the closed loop runs at this
script's cadence (re-issue ``run_policy`` to change the goal). Headless-friendly
(no TTY/agent) so it doubles as the reproducible demo artifact.

Each segment records to its own MP4 beside ``--mp4`` (``<stem>.seg0.mp4`` ...),
and the segments are joined into ``--mp4`` with
:func:`strands_robots.rendering.concat_clips` once the schedule ends. A rollout
opens its video fresh, so handing every segment the same path would keep only
the last one - a two-second clip of the robot halting. Pass ``--keep-segments``
to leave the per-segment files in place.

The G1 scene's ``default`` camera frames the origin from a fixed vantage, so a
walking robot reaches the edge of the frame within a couple of seconds. The
script mounts a camera on the pelvis (``add_camera(parent_body=...)``) before
the first rollout - :data:`FOLLOW_CAMERA` - so the view rides with the robot
and every segment is recorded from it.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    # checkpoint dir with policy.onnx (+ walk_policy.onnx); see docs/policies/wbc.md
    MUJOCO_GL=egl python examples/locomotion/scripted_g1.py \
        --checkpoint /path/to/grootwbc-g1 --mp4 /tmp/g1_locomotion.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from strands_robots import Robot
from strands_robots.rendering import concat_clips

# A short steer: forward, veer left, forward faster, then halt. Each entry is
# (duration_s, policy_kwargs) - a plain goal dict forwarded verbatim to the
# policy. ``target_velocity`` is [vx, vy, wz] (planar velocity + yaw rate).
SCHEDULE: list[tuple[float, dict[str, list[float]]]] = [
    (2.0, {"target_velocity": [0.4, 0.0, 0.0]}),
    (2.0, {"target_velocity": [0.4, 0.0, 0.5]}),
    (2.0, {"target_velocity": [0.6, 0.0, 0.0]}),
    (2.0, {"target_velocity": [0.0, 0.0, 0.0]}),
]


#: A camera that rides on the pelvis, behind and to the right of the robot,
#: looking a little ahead of it. ``position``/``target`` are in the pelvis frame
#: (x forward), so it follows the walk and turns with the veer; the base's
#: pitch and roll while walking are small enough to read as a hand-held follow.
FOLLOW_CAMERA: dict[str, Any] = {
    "name": "follow",
    "parent_body": "unitree_g1/pelvis",
    "position": [-2.6, -1.6, 1.1],
    "target": [0.4, 0.0, -0.3],
    "fov": 45,
}


def segment_path(mp4: str | Path, index: int) -> Path:
    """The per-segment clip for segment ``index`` of the run that ends in ``mp4``."""
    out = Path(mp4)
    return out.with_name(f"{out.stem}.seg{index}{out.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--mp4", default="/tmp/g1_locomotion.mp4")
    parser.add_argument(
        "--keep-segments",
        action="store_true",
        help="leave the per-segment clips (<stem>.seg<i>.mp4) beside the joined --mp4",
    )
    args = parser.parse_args(argv)

    robot = Robot("unitree_g1", mode="sim")
    added = robot.add_camera(width=640, height=480, **FOLLOW_CAMERA)
    if added.get("status") != "success":
        print(f"add_camera: {added['content'][0]['text']}")
        return 1
    policy_config = {"checkpoint": args.checkpoint, "walk": True}
    status = 0
    segments: list[Path] = []
    for i, (duration, goal) in enumerate(SCHEDULE):
        # One short-horizon segment per goal, each to its own clip: a rollout
        # opens its video fresh, so one shared path would keep only the last.
        segment = segment_path(args.mp4, i)
        result = robot.run_policy(
            robot_name="unitree_g1",
            policy_provider="wbc",
            policy_config=policy_config,
            policy_kwargs=goal,
            duration=duration,
            control_frequency=50.0,
            video={"path": str(segment), "fps": 30, "camera": FOLLOW_CAMERA["name"], "width": 640, "height": 480},
        )
        print(f"segment {i} goal={goal}: {result['content'][0]['text']}")
        if result.get("status") != "success":
            status = 1
            break
        segments.append(segment)
    if segments:
        joined = concat_clips(segments, args.mp4)
        print(f"joined {len(segments)} segment(s) into {joined}")
        if not args.keep_segments:
            for segment in segments:
                segment.unlink(missing_ok=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
