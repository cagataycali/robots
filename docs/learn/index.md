---
description: One page per workflow question, each ending with something that runs; pick the row that matches what you are doing.
---

# Learn

| you are | start here | then |
|---|---|---|
| running a learned policy | [Policies](policies/index.md): what runs where, the three known gaps, one page per provider | [lerobot_local](policies/lerobot-local.md) runs a Hub checkpoint in process |
| putting a robot in front of a model | [Agents](agents.md): the robot as a tool, the tools around it, the operator gate | [Security](security.md): every control between a model and a motor |
| building a scene | [Simulation](simulation/index.md): MuJoCo, Isaac, Newton, worlds, predicates, randomization | [Training](training/index.md): post-tuning with [LeRobot](training/lerobot.md), [RL](training/rl.md) |
| recording and training | [Record](data/record.md), [Verify](data/verify.md), [Foxglove](data/foxglove.md), [Label and judge](data/label-and-judge.md), [Stream and sync](data/stream-and-sync.md) | a dataset from the first frame to a filtered training set, and a live view while it records |
| wiring hardware | [Drivers](hardware/drivers.md): the contract and the shipped drivers; [Teleoperation](hardware/teleoperation.md), [Cameras](hardware/cameras.md), [Calibration](hardware/calibration.md) | setup pages: [Feetech arms](hardware/feetech-arms.md), [Unitree](hardware/unitree.md), [Franka](hardware/franka.md), [UR](hardware/ur.md), [Reachy Mini](hardware/reachy-mini.md), [Microduck](hardware/microduck.md), [Booster T1](hardware/booster-t1.md) |
| operating several robots | [Mesh](mesh/index.md): what it is and the three switches; [Fleet](mesh/fleet.md), [Safety and e-stop](mesh/safety-and-estop.md), [Topics](mesh/topics.md), [Bridges](mesh/bridges.md) | [Dashboard](dashboard.md), [ROS 2](ros2.md): three transports, one gate |

Every runnable code block on these pages was executed against this commit. A block marked `sketch` needs hardware, a GPU or an account, and says which.
