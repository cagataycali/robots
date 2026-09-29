---
description: flux3_action runs Black Forest Labs' FLUX 3 Action SO-101 checkpoint in process and drives the arm in MuJoCo or on hardware at 30 Hz.
---

# flux3_action

Drive the MuJoCo SO-101 with `black-forest-labs/flux-3-action-so101` from one `run_policy` call and record a LeRobot dataset.

```bash
pip install 'strands-robots[sim-mujoco,lerobot]'
pip install 'flux-action[encoders] @ git+https://github.com/black-forest-labs/flux-action'
pip install natten==0.21.6 -f https://whl.natten.org   # the wheel for your torch + CUDA build
```

`flux-action` is git-only, so the `flux3` extra is a placeholder.

## What it is

`Flux3ActionPolicy` wraps the FLUX 3 Action inference library (Apache 2.0): a 3.9B flow-matching transformer over a video VAE that reads eight steps of two camera views plus joint state and an instruction, returning 42 absolute targets for the six SO-101 joints, replanning every 32 ticks at 30 Hz. Weights load in process on `load()`, the first `reset()` or `get_actions()`.

## Constructor keywords

{{providers:kwargs:flux3_action}}

`mode="queued"` (default) is the library's `select_action` loop: 32 of the 42 steps are queued and the model replans when the queue empties. `mode="chunk"` calls the stateless `predict_action_chunk` once per `get_actions` and returns the whole chunk.

## Unit frame

The checkpoint speaks lerobot SO-101 units, arm degrees and gripper percent, in the frame of the community episodes it was tuned on (rest about `(0, 190, 180, 70, 0)`). The MuJoCo `so101` model reports radians around a different zero with `shoulder_lift` reversed. `UnitAdapter` converts both directions; `joint_units`, `joint_signs`, `joint_offsets_deg` and `gripper_range` default to the MuJoCo calibration read off the model's forward kinematics; a lerobot-calibrated real arm takes `joint_units="deg"`, unit signs, zero offsets and `gripper_range=(0, 100)`. The provider warns once when a converted state leaves the checkpoint's q01..q99 window. `SO101_SIM_REST_QPOS_RAD` is the folded rest posture the training episodes start from.

## Cameras

Two views, matched by name: `scene` (fixed) and `wrist` (on `so101/gripper`), any RGB size, resized to 256 by 256; a missing view is a `ValueError` naming it.

## Run it

```python title="sketch"
from strands_robots import Robot

robot = Robot("so101", mode="sim")
robot.add_camera(name="scene", position=[0.30, -0.62, 0.32], target=[0.0, -0.15, 0.06], width=256, height=256)
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.06, 0.0, 0.0], target=[-0.008, 0.0, -0.16], width=256, height=256)
robot.add_object(name="cube", shape="box", position=[0.0, -0.20, 0.02], size=[0.02, 0.02, 0.02], color=[0.9, 0.1, 0.1, 1])
robot.start_recording(repo_id="you/f3a-so101-mujoco", task="pick up the red cube", fps=30, cameras=["scene", "wrist"])
robot.run_policy(
    robot_name="so101",
    policy_provider="flux3_action",
    policy_config={"device": "cuda"},
    instruction="pick up the red cube",
    duration=15.0,
    control_frequency=30,
    n_episodes=10,
    reset_between=True,
)
robot.stop_recording()
```

`examples/vla/flux3_action_so101_sim.py` is this loop, resetting to rest.

## Jetson Thor

The published NATTEN wheels carry kernels for sm_75 through sm_120 but not sm_110, so on Thor the default `cutlass-fna` dies mid-rollout with `no kernel image is available`. The provider reads the arch list off `libnatten` with `cuobjdump`, selects `flex-fna` when this device is missing (`natten_backend` or `F3_NATTEN_BACKEND` override it) and forwards the choice to `na2d` / `na3d` as `backend=`, which `flux_action` alone does not. Measured on Thor: 24 s warm load, about 5 s per replan, so sim runs about 8x slower than wall clock.

## Limits

- SO-101 only; the DROID checkpoint (7 joints in radians plus gripper at 15 Hz) is not mapped.
- About 22 GB of GPU memory in bf16; 14 GB of weights.
- The shipped `so101` asset cannot hold a cube by friction (issues #2145 and #2167); scripted picks weld it with `attach_bodies`, a policy rollout cannot, so judge a sim episode on the reach and grasp.
