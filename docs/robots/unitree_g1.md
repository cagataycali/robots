---
title: unitree_g1
description: "Unitree G1 Humanoid (29-DOF + dexterous hands)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree G1 Humanoid (29-DOF + dexterous hands)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">46 joints</span><span class="sr-chip sr-chip-sim">sim</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot, strands</span></p>

<robot-viewer name="unitree_g1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_g1")
```

```python title="sketch"
robot = Robot("unitree_g1", mode="real", driver="lerobot", port="/dev/ttyACM0")  # lerobot unitree_g1
robot = Robot("unitree_g1", mode="real", port="192.168.123.164", network_interface="eth0")  # G1Driver
```

Aliases: `g1`, `g1_wbc`, `real_g1_relative_eef_relative_joints`, `unitree_g1_full_body`, `unitree_g1_locomanip`, `unitree_g1_real`, `unitree_g1_sonic`, `unitree_g1_wbc`.

## Hardware

**lerobot.** `Robot("unitree_g1", mode="real")` builds lerobot's `unitree_g1` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict.

**`G1Driver`** (the default for this robot) speaks CycloneDDS through `unitree_sdk2py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#g1driver).

## Policies verified on this robot

| Checkpoint | Provider | Where | What happened |
|---|---|---|---|
| NVlabs GR00T-WholeBodyControl G1 (two ONNX files, local directory) (whole-body controller, SONIC) | `wbc` | sim, laptop CPU | 100 steps at 50 Hz in 2.0 s, 1.7 ms inference, the G1 walked 0.665 m and stayed upright (pelvis 0.793 to 0.747 m); drives 15 of 29 actuators by design, so partial_action_failure_rate reads 0.48 on a healthy rollout. A HuggingFace id is refused (#4161): pass the local directory. Source: interface sweep 2026-09-28 script sim-policies/06b and issue #4161 |
| amazon-far/holosoma fastsac_g1_29dof.onnx (Apache-2.0) (whole-body controller, Holosoma) | `holosoma` | sim, laptop CPU | 0.5 m/s for 5 s at 50 Hz: walked 1.892 m with 0.036 m of drift, pelvis 0.789 m, real time on the CPU; ppo 1.721 m; wbc on the same scene 1.888 m. Drives all 29 actuators. Source: PR #4249 body, rollouts 2026-09-29 |

Providers written for this body: `wbc`, `holosoma`, `wbc_gait`, `kimodo`, `protomotions`; the rest are in the [policy matrix](../learn/policies/index.md).

Model: [google-deepmind/mujoco_menagerie/unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_g1), scene `scene.xml`.
