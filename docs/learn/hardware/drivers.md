---
description: How Robot(name, mode="real") picks its driver, the native drivers that ship, and the contract a driver implements.
---

# Drivers

How `Robot(name, mode="real")` picks a driver, the {{n:native_drivers}} native drivers that ship and their wires, and the contract a driver implements.

```python
from strands_robots.drivers import list_native_drivers, list_driver_coverage

print(list_native_drivers()["so101"])            # FeetechDriver
print(list_driver_coverage()["so101"])           # ('lerobot', 'strands')
print(list_driver_coverage()["omx"])             # ('lerobot',)
print(list_driver_coverage()["ability_hand"])    # ()
```

## Two driver families

| `driver=` | builds | for |
|---|---|---|
| `"strands"` (`NATIVE_DRIVER`, what `auto` picks when one is registered) | the native driver class registered for the robot | the robots in the table below |
| `"lerobot"` (`DEFAULT_DRIVER`, what `auto` falls back to) | `strands_robots.hardware_robot.Robot` around a lerobot robot class | any robot whose registry entry has `hardware.lerobot_type` (`so101_follower`, `koch_follower`, `lekiwi`, `bi_so_follower`, ...) |
| `"auto"` (the default) | the registry's `hardware.driver` if set, else a registered native driver, else lerobot | everything |

Robots lerobot has no type for (`unitree_go2`, `robotiq_2f85`, `reachy_mini`, `microduck`, `booster_t1`, `crazyflie`, `yahboom_m3pro`) declare `hardware.driver = "strands"`; other robots in the table below need none: `Robot("so101", mode="real", port="/dev/ttyACM0")` builds `FeetechDriver` with no lerobot extra, with the arm's lerobot calibration. `omx`, `openarm` and `reachy2` have no native driver and fall back to lerobot; `driver="lerobot"` pins that path, and `earthrover` declares it for teleop reads. `driver="strands"` on a robot with no native driver is refused by name.

`port=` is a Feetech serial path, a controller IP, a `radio://` URI for a Crazyflie, `host:port` for a daemon. A keyword the driver does not declare is refused.

## Shipped native drivers

Generated from `_SHIPPED_DRIVERS` and each module's `SUPPORTED_ROBOTS`:

{{drivers_table}}

{{driver_facts}}

Every native driver imports its SDK in `connect_eagerly()`, so a missing SDK is a refusal naming the install line.

## The contract

A native driver is anything with these members (`HardwareDriver` is a `runtime_checkable` Protocol; no inheritance needed):

| member | role |
|---|---|
| `tool_name`, `tool_type`, `tool_spec`, `stream` | the Strands `AgentTool` surface, so `Agent(tools=[robot])` works |
| `send_action(action, robot_name=None)` | one command, keyed by this driver's joint names; returns a status envelope |
| `run_policy(policy, ...)`, `get_task_status()`, `stop_task()`; `start_task(instruction, policy_provider=...)` builds the policy in the driver and is removed in 0.8 | the policy rollout path |
| `get_status()` (async), `stop()` (async) | health and de-energise |
| `cleanup()` | release the transport |

Constructor: `driver_cls(tool_name=..., cameras=..., data_config=..., **kwargs)`. A driver that wants a `cameras=` dict sets `reads_cameras = True`; otherwise a non-empty `cameras=` is refused, not dropped.

Optional: `get_observation` and the sensor attributes (`_pose`, `_imu`, `_battery`, `_lidar_state`), which the mesh reads with `getattr`, so a driver without an IMU publishes no IMU topic. Joint telemetry needs a `bus` with `sync_read` or a `get_observation`, plus `is_connected`.

A refusal has the same envelope shape as a success:

```python
from strands_robots import Robot

sim = Robot("so101")
arm = Robot("so101", mode="real", driver="strands", transport="twin", sim=sim)
arm.connect_eagerly()
print(arm.send_action({"shoulder_pan": 10.0}))
# {'status': 'success', 'content': [{'json': {'commanded': {'shoulder_pan': 10.0}, 'unit': 'degrees (gripper: percent open)'}}]}
arm.cleanup()
```

The real `FeetechDriver`, its MuJoCo model on the bus: verbs, units and refusals without a serial port.

## Register your own

```python title="sketch"
from strands_robots.drivers import register_native_driver

register_native_driver("koch_follower", MyKochDriver)   # refuses a class missing a contract member
robot = Robot("koch_follower", mode="real", driver="strands", port="/dev/ttyUSB0")
```

`register_native_driver` binds a driver class to a registry robot name, as its default, after `missing_driver_members(cls)` passes; re-registering needs `overwrite=True`. For an unregistered robot, call `register_robot("my_arm", model_xml=..., hardware={"driver": "strands"})` first.

## Where the gates are

A driver refuses before writing: the Feetech bus past servo travel, the G1 outside its FSM handshake or under 15% battery, the Go2 before sport release, the Booster T1 before upper-body control, the Robotiq before activation, the UR in `PROTECTIVE_STOP`, the xArm or Gen3 on a fault, the iiwa outside FRI commanding, the Stretch past SDK clipping. Above them sits [the operator gate](../agents.md#the-operator-gate).

Next: [feetech-arms](feetech-arms.md), [teleoperation](teleoperation.md), [cameras](cameras.md), [calibration](calibration.md).
