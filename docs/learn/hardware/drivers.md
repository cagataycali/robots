---
description: How Robot(name, mode="real") picks its driver, the native drivers that ship, and the contract a driver implements.
---

# Drivers

At the end of this page you know how `Robot(name, mode="real")` picks the code that talks to your robot, which of the {{n:native_drivers}} native drivers ship and over which wire, and what a driver must implement for the agent, the mesh and the teleop loop.

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
| `"lerobot"` (`DEFAULT_DRIVER`, what `auto` falls back to) | `strands_robots.hardware_robot.Robot` around a lerobot robot class | any robot whose registry entry has `hardware.lerobot_type` (`so101_follower`, `koch_follower`, `lekiwi`, `bi_so_follower`, ...) |
| `"strands"` | the native driver class registered for the robot | the robots in the table below |
| `"auto"` (the default) | the registry's `hardware.driver` if set, else lerobot | everything |

Robots lerobot has no type for (`unitree_go2`, `robotiq_2f85`, `reachy_mini`, `microduck`, `booster_t1`, `crazyflie`, `yahboom_m3pro`) declare `hardware.driver = "strands"` in the registry, so a bare `Robot("unitree_go2", mode="real", port="192.168.123.161")` builds the native driver. `panda` and `ur5e` have native drivers but no `hardware` block, so they resolve to lerobot until you ask: `Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")`. Asking for `driver="strands"` on a robot with no native driver is refused by name, never served the lerobot path quietly.

`port=` is polymorphic: a serial path for a Feetech bus, an IP for a controller, a `radio://` URI for a Crazyflie, `host:port` for a daemon. Each driver documents what it reads. A keyword the driver does not declare is refused (`Robot(..., prot="/dev/ttyACM0")` does not build an arm that auto-detects a port).

## Shipped native drivers

Generated from `_SHIPPED_DRIVERS` and each module's `SUPPORTED_ROBOTS`:

{{drivers_table}}

{{driver_facts}}

Every native driver imports its SDK inside `connect_eagerly()`, never at module import, so the package imports on a machine without the SDK and a missing SDK is a named refusal with the install line in it.

## The contract

A native driver is anything with these members (`HardwareDriver` is a `runtime_checkable` Protocol; no inheritance needed):

| member | role |
|---|---|
| `tool_name`, `tool_type`, `tool_spec`, `stream` | the Strands `AgentTool` surface, so `Agent(tools=[robot])` works |
| `send_action(action, robot_name=None)` | one command, keyed by this driver's joint names; returns a status envelope |
| `run_policy(policy, ...)`, `get_task_status()`, `stop_task()`; `start_task(instruction, policy_provider=...)` builds the policy in the driver and is removed in 0.8 | the policy rollout path |
| `get_status()` (async), `stop()` (async) | health and de-energise |
| `cleanup()` | release the transport |

Constructor: `driver_cls(tool_name=..., cameras=..., data_config=..., **kwargs)`. A driver that wants a `cameras=` dict sets `reads_cameras = True`; otherwise a non-empty `cameras=` is refused rather than silently dropped.

Deliberately absent: `get_observation` and the sensor attributes (`_pose`, `_imu`, `_battery`, `_lidar_state`). The mesh reads them with `getattr(robot, name, None)`, so a driver without an IMU publishes no IMU topic and is otherwise complete. Joint telemetry reaches the mesh when a driver exposes either a `bus` with `sync_read` or a `get_observation`, plus `is_connected`.

Every refusal returns the same envelope shape as a success:

```python
from strands_robots import Robot

sim = Robot("so101")
arm = Robot("so101", mode="real", driver="strands", transport="twin", sim=sim)
arm.connect_eagerly()
print(arm.send_action({"shoulder_pan": 10.0}))
# {'status': 'success', 'content': [{'json': {'commanded': {'shoulder_pan': 10.0}, 'unit': 'degrees (gripper: percent open)'}}]}
arm.cleanup()
```

That is the real `FeetechDriver` with the arm's MuJoCo model at the far end of the bus (`transport="twin"`), the way to exercise a native driver's verbs, units and refusals without a serial port.

## Register your own

```python title="sketch"
from strands_robots.drivers import register_native_driver

register_native_driver("koch_follower", MyKochDriver)   # refuses a class missing a contract member
robot = Robot("koch_follower", mode="real", driver="strands", port="/dev/ttyUSB0")
```

`register_native_driver` binds a driver class to a registry robot name after `missing_driver_members(cls)` passes; double registration is refused unless `overwrite=True`. It does not make a new name known: for a robot the registry does not know, call `register_robot("my_arm", model_xml=..., hardware={"driver": "strands"})` first, then register the driver under that name. A package outside this repo registers at import time; the shipped table tolerates a caller registering first.

## Where the gates are

A driver refuses before it writes: the Feetech bus refuses a target outside the servo's travel, the G1 refuses outside its FSM handshake states or under 15 percent battery, the Go2 refuses until sport mode is released, the Booster T1 refuses until upper-body control is enabled, the Robotiq refuses until activation completes, the UR refuses in `PROTECTIVE_STOP`. Those are facts about the hardware; the operator approval above all of them is [the operator gate](../agents.md#the-operator-gate).

Next: [feetech-arms](feetech-arms.md), [teleoperation](teleoperation.md), [cameras](cameras.md), [calibration](calibration.md).
