---
title: unitree_go2
description: "Unitree Go2 Quadruped"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree Go2 Quadruped

{{robot_chips:unitree_go2}}

<robot-viewer name="unitree_go2"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_go2", position=[0.0, 0.0, 0.003])
print(robot.robot_action_keys("unitree_go2"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

```python title="sketch"
robot = Robot("unitree_go2", mode="real", port="192.168.123.161", network_interface="eth0")  # Go2Driver
```

Aliases: `go2`.

## Hardware

**`Go2Driver`** (the default for this robot) speaks CycloneDDS through `unitree_sdk2py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#go2driver).

## Policies verified on this robot

| Checkpoint | Provider | Where | What happened |
|---|---|---|---|
| none: builtin benchmark go2_walk_forward with the mock policy (benchmark plumbing, not a trained policy) | `mock` | sim, laptop CPU | evaluate_benchmark ran one episode in 1.19 s and ended it at step 27 of 1000 on the failure clause (the test motion tips the robot), avg_reward 0.49, pass 0 of 1. The scene robot must be spelled unitree_go2: the alias go2 is refused by the benchmark (#4160). Source: interface sweep 2026-09-28 scripts sim-policies/04 and 04b and issue #4160 |

Model: [google-deepmind/mujoco_menagerie/unitree_go2](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_go2), scene `scene.xml`.
