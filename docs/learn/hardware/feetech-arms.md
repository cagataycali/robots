---
description: An SO-100 or SO-101 on a serial port: which driver talks to it, the units a command takes, checking the bus.
---

# Feetech arms

At the end of this page an SO-100 or SO-101 (and the LeKiwi it rides on) is open on a serial port, you know which of the two drivers is talking to it, what units a command takes, and how to check the bus before you trust a policy with it. Koch arms are Dynamixel and appear only to say where they go.

This needs an arm on USB. Find the port first:

```bash
python -m serial.tools.list_ports -v          # /dev/ttyACM0 on Linux, /dev/tty.usbmodem* on macOS
```

```python title="sketch"
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/ttyACM0")                     # lerobot driver, the default
arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0")   # native FeetechDriver
```

## Which driver

| | `driver="lerobot"` (default) | `driver="strands"` |
|---|---|---|
| class | `hardware_robot.Robot` around lerobot `so101_follower` | `drivers.feetech.FeetechDriver` |
| needs | `pip install 'strands-robots[lerobot]'` | `pip install pyserial` |
| units in `send_action` | degrees (`so101_follower` sets `use_degrees=True`); `gripper.pos` 0 to 100 | degrees; `gripper` is percent open |
| keys | `shoulder_pan.pos` | `shoulder_pan` or `shoulder_pan.pos`, one per motor |
| policy rollout | yes (`execute`, `start`) | yes, `PolicyRollout` at `control_frequency` 30 Hz |
| cameras | `cameras={...}` opened by lerobot | not read (`reads_cameras` is not set) |
| teleop leader | `Teleoperator("so101_leader", port=...)` | same, through `attach_teleop` |

Both register for `so100`, `so101`, `lekiwi`, `hope_jr` and `open_duck_mini`. `hope_jr` and `open_duck_mini` share the bus protocol but not the six-servo layout; pass `motor_ids=` to the native driver until a joint map for them lands.

The native driver's six motors, servo ids 1 to 6 in wire order from `shoulder_pan` to `gripper`, are the generated joint table on the [so101 page](../../robots/so101.md); five take degrees, the gripper 0 to 100 percent open.

## Check the bus without moving

`serial_tool` reads and pings without approval; only `send`, `send_read`, `feetech_position` and `feetech_velocity` are gated.

```python title="sketch"
from strands_robots import serial_tool

serial_tool(action="list_ports")
serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
```

On the lerobot path the robot tool's `get_state` reads positions, torque and voltage over the bus lock without calling lerobot's `connect()` (which would write PID registers). An uncalibrated arm reports ticks converted with `(ticks - 2048) * 360 / 4096` and says so.

## Move one joint

Through the agent, `pose_tool` names poses and moves motors; every motion verb is gated (see [agents](../agents.md)). Direct from Python on the native driver:

```python title="sketch"
arm.send_action({"shoulder_pan": 0.0, "gripper": 50.0})
import asyncio; asyncio.run(arm.stop())        # releases torque on every motor
```

`stop()` de-energises. `stop_task()` only halts a rollout and leaves the arm holding position.

## Calibration

Run `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_arm` once. The lerobot driver reads that file by `id`; the native driver takes `calibration=` (the path, or the records) and otherwise commands the servo's full travel and reports `calibration_source` in `get_status()`. Details in [calibration](calibration.md).

## LeKiwi

`lekiwi` is an SO-101 on a three-wheel holonomic base with a Raspberry Pi. lerobot's `lekiwi` type runs on the Pi and `lekiwi_client` on your laptop; `Robot("lekiwi", mode="real", robot_ip="192.168.1.50")` builds the client. The native `FeetechDriver` registers for it too and drives the arm servos over a local serial port.

## Twin transport

The native driver runs unchanged against the arm's MuJoCo model, which is how the [drivers](drivers.md) page exercises it without hardware:

```python
from strands_robots import Robot

sim = Robot("so101")
arm = Robot("so101", mode="real", driver="strands", transport="twin", sim=sim, realtime=False)
arm.connect_eagerly()                                  # None on success, a reason string otherwise
arm.send_action({"gripper": 100.0})                    # percent open
for _ in range(20):
    sim.step()
print(arm.bus.sync_read("Present_Position")["gripper"])  # ~99.9, read back from the MuJoCo joint
arm.cleanup()
```

## Koch and other Dynamixel arms

`Robot("koch", mode="real", driver="strands", port=...)` builds `DynamixelDriver`: the verbs, units and refusals above over a Protocol 2.0 bus, with the `koch_follower` calibration file; without `driver=`, koch resolves to lerobot's `koch_follower`. ViperX, WidowX and ALOHA have no verified motor map and are refused.
