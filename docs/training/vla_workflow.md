---
description: VLA workflow on the Unitree G1 - collect teleop data in sim, then deploy a SONIC whole-body-control checkpoint.
---

# VLA-on-G1 Workflow

The Vision-Language-Action (VLA) pipeline on the Unitree G1 humanoid:
**collect teleop data** (LeRobot recording) -> **deploy with SONIC whole-body
control** (WBC provider). Post-training the base VLA between those two stages is
run with upstream Isaac-GR00T's own tooling; this package does not vendor that
trainer, and the checkpoint it produces deploys through the WBC provider
unchanged.

Each piece ships individually in `strands-robots`; this page documents how they
compose into one coherent pipeline. The companion example script runs the chain
end-to-end:

```bash
# Quick demo (record + deploy with mock, ~10s on CPU):
python examples/locomotion/vla_g1_workflow.py

# Deploy-only with downloaded SONIC weights:
python examples/locomotion/vla_g1_workflow.py --checkpoint /path/to/grootwbc-g1
```

## Pipeline stages

### 1. Record  - collect locomotion data

Drive the G1 (in sim or on real hardware via LeRobot teleop) and capture a
`LeRobotDataset`. The recording pipeline is the same one the existing
[`03_record_dataset.py`](https://github.com/strands-labs/robots/blob/main/examples/03_record_dataset.py)
hero example demonstrates  - adapted for the 29-DOF humanoid:

Drive the G1 with **WBC** (the merged SONIC whole-body controller) so the
captured dataset is genuine walking motion:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy
from strands_robots.policies.wbc import install_wbc_torque_control

sim = Robot("unitree_g1", mesh=False)
policy = create_policy("wbc", checkpoint="/path/to/grootwbc-g1", walk=True)

# WBC emits joint-POSITION targets, but the G1 scene's actuators are
# position-servos (uniform kp=500) that override SONIC's tuned per-joint PD -
# so writing the targets directly makes the robot fall. install_wbc_torque_control
# flips the G1's actuators to torque mode and applies the SONIC PD law at the
# correct decimation, so the G1 actually WALKS. Pair it with control_frequency=50.
install_wbc_torque_control(sim, policy, "unitree_g1")

sim.start_recording(
    repo_id="local/g1_locomotion",
    root="/tmp/g1_dataset",
    # Matches control_frequency=50.0 below - the dataset rate IS the capture rate.
    fps=50, task="walk forward", overwrite=True,
)
sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    instruction="walk forward",
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},  # [vx, vy, omega]
    action_horizon=1,
    control_frequency=50.0,
    n_steps=200,
)
sim.stop_recording()
```

The `vla_g1_workflow.py` example wires exactly this up behind a flag:

```bash
python examples/locomotion/vla_g1_workflow.py --record-checkpoint /path/to/grootwbc-g1
```

Two ingredients make WBC close its loop through `sim.run_policy`:

1. The MuJoCo backend's observation surfaces the joint velocities and base IMU
   signals (`<joint>.vel`, `base_quat`, `base_ang_vel`) that WBC's balance
   controller consumes - no manual observation wiring.
2. `install_wbc_torque_control` converts WBC's position targets into joint
   torques via the SONIC PD law on torque-mode actuators (the standard scene
   ships stiff position-servos that the gait cannot drive).

For data collection from a different source, swap the WBC policy for a LeRobot
teleop driver, a VR controller, or `MockPolicy` (synthetic, runs with no weights
or hardware - the quick-demo default). The dataset format is identical either way.

To train a **language-conditioned (steerable)** policy, annotate the recorded
dataset with language columns first - see [Steerable annotation](../data/annotation.md).

### 2. Post-train (outside this package)

Post-training a base VLA on the recorded dataset is run with the upstream
Isaac-GR00T tooling linked below. The recorded dataset is a standard
`LeRobotDataset`, so it is consumed as-is, and the checkpoint that comes out is
what stage 3 deploys.

For arm manipulation the equivalent step *is* in-package: the
[`Trainer` abstraction](overview.md) post-tunes a LeRobot policy on a recorded
dataset - see [`07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py).

### 3. Deploy  - SONIC whole-body control


Load the fine-tuned (or pre-trained SONIC) checkpoint with the `wbc` provider
and drive the G1's 15 leg+waist DOFs:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

sim = Robot("unitree_g1", mesh=False)
policy = create_policy("wbc", checkpoint="/tmp/g1_finetuned", walk=True)

sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    instruction="walk forward",
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
    action_horizon=1,
)
```

For real deploy-grade locomotion (with the upstream torque-PD law), use the
[torque-control harness](../policies/wbc.md#watching-it-walk-torque-control-deploy):

```bash
python examples/wbc/wbc_g1_torque_deploy.py --checkpoint /tmp/g1_finetuned --vx 0.5
```

## Prerequisites

| Stage | Install | External |
|-------|---------|----------|
| Record | `pip install "strands-robots[sim-mujoco,lerobot]"` | None (sim) |
| Post-train | upstream Isaac-GR00T tooling | Docker + GPU |
| Deploy | `pip install "strands-robots[wbc,sim-mujoco]"` | None (CPU ONNX) |

## Upstream references

- [GR00T Whole-Body-Control VLA workflow tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)
- [GR00T-WholeBodyControl repo](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [WBC policy docs](../policies/wbc.md)
- [Training overview](overview.md) (the `Trainer` abstraction)
- [Dataset recording example](https://github.com/strands-labs/robots/blob/main/examples/03_record_dataset.py)

## See also

- [`07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py)  - the same record->train->deploy loop for arm manipulation (SO-100 + LeRobot ACT)
- [WBC provider](../policies/wbc.md)  - the deploy-stage policy (observation layout, command kwargs, torque harness)
