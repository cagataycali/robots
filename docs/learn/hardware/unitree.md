---
description: Install the Unitree SDK the way that works, reach a G1 or Go2 over CycloneDDS, and the gate each driver enforces before a motor frame.
---

# Unitree G1 and Go2

At the end of this page the vendor SDK is installed the one way that works, `Robot("g1", mode="real")` or `Robot("unitree_go2", mode="real")` reaches the robot over CycloneDDS, you know which safety gate each driver enforces before it writes a motor frame, and which agent verbs sit on top.

This needs the robot on the same Ethernet segment and the SDK below. Nothing in the package imports `unitree_sdk2py` at module load; a missing SDK is a refusal with this recipe in it.

```python title="sketch"
from strands_robots import Robot

g1 = Robot("g1", mode="real", driver="strands", port="192.168.123.161", network_interface="eth0")
go2 = Robot("unitree_go2", mode="real", port="192.168.123.161", network_interface="eth0")
print(g1.connect_eagerly())          # None, or a reason string
```

## Install the SDK

`unitree_sdk2py` is not an extra of this package because the PyPI `unitree-sdk2` wheel lacks its `g1` and `comm` packages and pins `cyclonedds==0.10.2`, which has no Python 3.12 wheel. The `[ros2]` extra carries the `cyclonedds` binding at the range this project bounds; the SDK comes from the vendor checkout. On macOS arm64 and x86_64 Linux:

```bash
pip install 'strands-robots[ros2]'
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python          # --no-deps skips the ==0.10.2 pin
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('ok')"
```

On Linux aarch64 (the Jetson on the robot) `cyclonedds` has no wheel, so build the C library first:

```bash
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_BUILD_TYPE=Release && sudo cmake --build /tmp/cdds/build --target install
export CYCLONEDDS_HOME=/usr/local
pip install 'cyclonedds==0.10.2'
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python
```

A partial install (bindings and IDL present, no `comm`) lets `connect_eagerly()` succeed and fails only at the motion switcher; the G1 reports it as `motion_switcher_open_error` in `get_status()`, the Go2 as the refusal from `release_sport_mode()`. Point `CYCLONEDDS_URI` at the robot's `cyclonedds.xml` when multicast discovery does not find it.

## Two drivers, two gates

| | `G1Driver` | `Go2Driver` |
|---|---|---|
| IDL | `unitree_hg.msg.dds_.LowCmd_` | `unitree_go.msg.dds_.LowCmd_` |
| reads | `rt/lowstate`, `rt/lf/bmsstate`, `rt/utlidar/lidar_state`, `rt/utlidar/cloud_livox_mid360`, `rt/mainboardstate`, `rt/pressuresensorstate` | `rt/lowstate`, `rt/lf/bmsstate` |
| write gate | FSM id in `HANDSHAKE_FSMS` `{500, 501, 801}` and battery at or above 15 percent | sport mode released (`CheckMode()` name is empty) and battery at or above 15 percent |
| unlock | motion switcher | `go2.release_sport_mode()` |
| control loop | `run_policy` at 500 Hz, per-step FSM re-gate, zero-torque frame on exit | `run_policy` at 500 Hz |
| joints | 29, by name in `g1.py` | 12, by name in `GO2_JOINT_INDEX`; an index is never accepted because the SDK's `LegID` order differs from the URDF order |

Both refuse rather than warn: publishing `rt/lowcmd` while the onboard controller still holds the motors means two controllers fighting over one robot. `send_action` takes joint targets keyed by name; a frame arrives at the motors only after the gate passes.

## Agent verbs

`use_unitree(service_name, operation_name, parameters)` wraps every SDK client (`loco`, `arm`, `audio`, `motion_switcher`, `vui`, `robot_state`) with dynamic discovery: `list_services`, `list_operations`, `describe_operation` work without the SDK by reading its source. Every write, and every `HIGH_DANGER_OPS` entry (`ZeroTorque`, `SetFsmId`, `SetVelocity`, `Move`, `ReleaseMode`), stops for operator approval; `STRANDS_UNITREE_COMMAND_ALLOW` takes `service.operation` entries or `*`. Private SDK names (`_Call`) are refused outright.

The `g1_*` verbs do work beyond one RPC: `g1_get_state` and `g1_sensor` read the driver's caches (battery, imu, lidar_state, lidar_summary, mainboard, pressure); `g1_send_action`, `g1_run_policy`, `g1_task` drive the gated control loop; `g1_set_fsm`, `g1_move_velocity`, `g1_stop_move`, `g1_set_stand_height`, `g1_set_swing_height`, `g1_balance_stand`, the `g1_safe_*` posture transitions and the gestures are the execution verbs; `g1_joints`, `g1_motion_gates`, `g1_arm_actions`, `g1_error_codes` are reference tables (`7401` is "Arm is holding - release first").

```python title="sketch"
from strands import Agent
from strands_robots.tools.g1 import g1_get_state, g1_move_velocity, g1_stop_move, use_unitree

agent = Agent(tools=[g1_get_state, g1_move_velocity, g1_stop_move, use_unitree])
agent("Stand up, walk forward half a metre, stop.")   # each motion pauses for your approval
```

## On the mesh

With `STRANDS_MESH=true` the G1 publishes `_imu`, `_battery`, `_lidar_state` and `_lidar_summary` from its caches at the mesh cadence, and `emergency_stop` from any peer reaches its `stop()`. See [safety and e-stop](../mesh/safety-and-estop.md).

## Simulation first

Both robots have MuJoCo assets: `Robot("g1")` and `Robot("unitree_go2")` build the twin, and a locomotion policy from [policies](../policies/index.md) runs there before it runs on the metal.
