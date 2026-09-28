#!/usr/bin/env python3
"""Laya (text-only typed decisions) as a System 1 gate on the so101 in MuJoCo.

Goal: Show the ``laya`` provider end to end: a privileged world reader supplies the cube / gripper poses that the
text state carries, Laya answers joint / direction / size (+ two calibration gates) per tick, one discrete
primitive is applied, and ``confidence_gate`` turns a low ``progress_ok`` probability into a hold you can audit
through ``policy.last_tick``.
Dependencies: pip install "strands-robots[sim-mujoco,laya]"  (Laya downloads ~1.7 GB of checkpoints on first use)
Expected output: per-tick lines "tick joint direction size gated latency" and a final distance to the cube.
Runtime: ~30 s on a GPU (first run adds the checkpoint download); CPU works at ~0.5 s per tick.
Zero-shot Laya has never seen a robot: this is a research baseline for the calibration study, not a controller.
"""

import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
import mujoco

from strands_robots import Robot, create_policy

robot = Robot("so101", mode="sim", mesh=False)
robot.add_object(name="cube", shape="box", position=[0.0, -0.22, 0.01], size=[0.02] * 3, color=[0.9, 0.1, 0.1, 1])
model = robot.mj_model
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "so101/gripper")
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")


def world():
    """What Laya may read besides the joints: fingertip site, cube body, contacts (privileged, sim only)."""
    data = robot.mj_data
    return {"gripper_xyz_m": data.site_xpos[site].tolist(), "cube_xyz_m": data.xpos[cube].tolist(), "contacts": []}


policy = create_policy(
    "laya", model="typed-decisions", questions_profile="joint_direction_size_gated", confidence_gate=0.5
)
policy.set_world_reader(world)


def observe(event):
    tick = policy.last_tick
    if type(event).__name__ == "RunPolicyStep" and tick is not None:
        applied = tick["applied"]
        print(
            f"tick {tick['tick']:3d} {tick['primitive']['joint']:>14} {'+' if tick['primitive']['direction'] > 0 else '-'} "
            f"{tick['primitive']['size']:<6} gated={tick['gated']!s:<5} p(progress)={tick['gate_value']:.2f} "
            f"applied={applied['joint']:<14} {tick['latency_ms']:.0f} ms"
        )


result = robot.run_policy(
    robot_name="so101",
    policy_object=policy,
    instruction="move the gripper to the red cube",
    n_steps=40,
    control_frequency=10,
    fast_mode=True,
    observer=observe,
)
distance = math.dist(world()["gripper_xyz_m"], world()["cube_xyz_m"])
print(f"status {result['status']}  final fingertip-to-cube distance {distance:.3f} m")
