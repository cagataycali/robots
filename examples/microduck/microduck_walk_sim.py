"""Drive the Pollen Microduck's walking policy in MuJoCo.

Loads Pollen's shipped ``alpha_walking.onnx`` into the native
:class:`~strands_robots.policies.microduck.MicroduckPolicy` and runs a real
MuJoCo rollout through the standard ``Robot(...).run_policy`` seam - the same
call any provider uses. The observation is fed RAW (the exported graph carries
its own input normaliser), and actions decode to
``motor_target = DEFAULT_POSE + action * action_scale``.

Run (from the repo root; a bare weight name is fetched from Pollen's Hub
repository ``pollen-robotics/microduck-policies`` on first use, a path is used
as is)::

    python examples/microduck/microduck_walk_sim.py --duration 8
    python examples/microduck/microduck_walk_sim.py --onnx /data/alpha_walking.onnx

Pass ``--vx`` to command a forward twist. No hardware required.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        default="alpha_walking.onnx",
        help="a local .onnx file, or the bare name of a weight in pollen-robotics/microduck-policies",
    )
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--control-frequency", type=float, default=50.0)
    ap.add_argument("--vx", type=float, default=0.3, help="forward twist command (m/s)")
    args = ap.parse_args()

    robot = Robot("microduck")
    robot.reset()

    policy = MicroduckPolicy(onnx_path=args.onnx)
    result = robot.run_policy(
        policy_object=policy,
        control_frequency=args.control_frequency,
        duration=args.duration,
        policy_kwargs={"target_velocity": [args.vx, 0.0, 0.0]},
    )
    print(f"Rollout done: {result}")


if __name__ == "__main__":
    main()
