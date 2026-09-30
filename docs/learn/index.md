---
description: One page per workflow question, each ending with something that runs; pick the row that matches what you are doing.
---

# Learn

Guides by subsystem, each ending in something that runs. New here? The [Start](../start/index.md) ladder orders these for a first week, and [Concepts](../concepts/index.md) explains the three objects every page uses.

| you want to | read |
|---|---|
| hand a robot to an agent and see the gate work | [Agents](agents.md) |
| build a scene, add objects and cameras, randomize | [Simulation](simulation/index.md) |
| run a Hub checkpoint, pick a provider | [Policies](policies/index.md) |
| record a dataset and train from it | [Record](data/record.md), [Training](training/index.md) |
| plug in a real arm | [Drivers](hardware/drivers.md), [Feetech arms](hardware/feetech-arms.md) |
| several robots, one e-stop | [Mesh](mesh/index.md), [Safety and e-stop](mesh/safety-and-estop.md) |

## Policies

- [Policies](policies/index.md): which provider for what you have, the provider matrix and one page per provider; [lerobot_local](policies/lerobot-local.md) runs a Hub checkpoint in process.

## Agents, simulation, training

- [Agents](agents.md): a `Robot` as a Strands tool, the tools around it, the operator gate, what a refusal looks like.
- [Simulation](simulation/index.md): MuJoCo, Isaac, Newton, worlds and objects, predicates and rollouts, randomization.
- [Training](training/index.md): what trains where; [LeRobot](training/lerobot.md), [RL](training/rl.md), [Isaac Lab](training/isaaclab.md).

## Data

- [Record](data/record.md), [Verify](data/verify.md), [Label and judge](data/label-and-judge.md), [Stream and sync](data/stream-and-sync.md): a dataset from the first frame to a filtered training set.

## Hardware

- [Drivers](hardware/drivers.md): the contract and the generated table of shipped drivers.
- [Teleoperation](hardware/teleoperation.md), [Cameras](hardware/cameras.md), [Calibration](hardware/calibration.md).
- Setup pages: [Feetech arms](hardware/feetech-arms.md), [Unitree](hardware/unitree.md), [Franka](hardware/franka.md), [UR](hardware/ur.md), [Reachy Mini](hardware/reachy-mini.md), [Microduck](hardware/microduck.md), [Booster T1](hardware/booster-t1.md).

## Mesh

- [Mesh](mesh/index.md): what it is and the three switches.
- [Fleet](mesh/fleet.md), [Safety and e-stop](mesh/safety-and-estop.md), [Topics](mesh/topics.md), [Bridges](mesh/bridges.md).

## Operate

- [Dashboard](dashboard.md): what `strands-robots dashboard` serves, who may click, the e-stop button.
- [ROS 2](ros2.md): three transports, when each, one gate.
- [Security](security.md): every control between a model and a motor, on one page.

Every runnable fence on these pages was executed against this commit. A fence marked `sketch` needs hardware, a GPU or an account, and says which.
