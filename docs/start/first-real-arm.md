---
description: Find the SO-101's USB port, rehearse the driver against the model, then move the physical arm through the native driver or lerobot.
---

# First real arm

At the end of this page you know which USB port your SO-101 is on, you have rehearsed the exact hardware driver against the arm's model, and you have the two lines that move the physical arm: one through the native Feetech driver, one through lerobot.

## Find the port

Plug the arm's controller board into USB. Then ask the host what it sees:

```python
from strands_robots._serial_discovery import scan_serial_devices, describe_serial_candidates

devices = scan_serial_devices()
for device in devices:
    print(device.port, device.stable_id, device.likely_servo_bus)
print(describe_serial_candidates(devices))
```

On a laptop with nothing plugged in you should see the built-in ports and a sentence saying none of them is a servo bus:

```text
/dev/cu.debug-console None False
/dev/cu.wlan-debug None False
/dev/cu.Bluetooth-Incoming-Port None False
None of this host's 3 serial device(s) looks like a servo bus: /dev/cu.debug-console, /dev/cu.wlan-debug, /dev/cu.Bluetooth-Incoming-Port.
```

With the arm attached one row reads `True`: `/dev/ttyACM0` on Linux, `/dev/cu.usbmodem...` on macOS. `stable_id` is the USB serial number, which survives a replug when the port path does not. On Linux, add yourself to the `dialout` group if the port exists but cannot be opened; `strands-robots doctor` checks this.

## Rehearse on the twin

The native driver has two transports: `serial`, the wire, and `twin`, the same driver with the arm's MuJoCo model at the far end of the bus. Same verbs, same units, same refusals. Run it before you touch the arm:

```python
import asyncio
from strands_robots import Robot

arm = Robot("so101", mode="real", driver="strands", transport="twin")
print(type(arm).__name__, arm.transport, arm.is_connected)
print(arm.connect_eagerly())          # None means the bus opened
print(arm.send_action({"shoulder_pan": 20.0, "elbow_flex": -30.0}))
print({k: round(v, 1) for k, v in arm.bus.sync_read("Present_Position").items()})
print(asyncio.run(arm.get_status())["content"][0]["json"]["motors"])
arm.cleanup()
```

You should see:

```text
FeetechDriver twin False
None
{'status': 'success', 'content': [{'json': {'commanded': {'shoulder_pan': 20.0, 'elbow_flex': -30.0}, 'unit': 'degrees (gripper: percent open)'}}]}
{'shoulder_pan': 20.1, 'shoulder_lift': 3.1, 'elbow_flex': -27.3, 'wrist_flex': 0.6, 'wrist_roll': 0.0, 'gripper': 9.1}
{'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3, 'wrist_flex': 4, 'wrist_roll': 5, 'gripper': 6}
```

In `mode="real"` the factory returns the driver, not a simulation. Targets are degrees, the gripper is percent open, and a key can be spelled `shoulder_pan` or `shoulder_pan.pos`. `sync_read` is the bus read a real arm answers with, here answered by the model.

## Move the arm

Two drivers can move an SO-101. Pick one per process.

**Native** (`driver="strands"`, no lerobot install, {{n:native_drivers}} drivers ship this way):

```python title="sketch"
import asyncio
from strands_robots import Robot
from strands_robots.drivers.feetech.bus import lerobot_calibration_path

arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0",
            calibration=lerobot_calibration_path("so101_follower", "so101"))
arm.connect_eagerly()
arm.send_action({"shoulder_pan": 20.0})
asyncio.run(arm.stop())                # releases torque; cleanup() alone leaves it held
arm.cleanup()
```

**lerobot** (`driver="lerobot"`, the default when the robot has no native driver, and the one that opens cameras):

```python title="sketch"
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/ttyACM0",
            cameras={"front": {"type": "opencv", "index_or_path": 0, "fps": 30}})
arm.send_action({"shoulder_pan.pos": 20.0})
arm.cleanup()
```

Both are the `so101` tool when handed to an agent. The lerobot path connects on the first action; the native path connects on `connect_eagerly()` or the first action. `cleanup()` closes the port and leaves torque as it is, so an arm holding a payload does not drop when a process exits; `stop()` is the verb that de-energizes.

## Calibrate

Calibration is the arm's measured travel per servo. Without it the driver reads and commands the servo's full rotation, which is off by however far the mechanical stops sit inside it. lerobot writes the file, and both drivers read it:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101
```

The lerobot driver looks the file up by id, and the id it uses is the tool name, `so101` unless you pass `tool_name=` or `id=`. The native driver takes the file as `calibration=`; `lerobot_calibration_path("so101_follower", "so101")` returns where lerobot put it, and `get_status()` reports `calibration_source` so you can tell whether the arm's travel or the servo's full rotation is in force. Details in [Calibration](../learn/hardware/calibration.md).

## What is refused before the arm moves

| you wrote | what happens |
|---|---|
| `Robot("so101", port="/dev/ttyACM0")` | `TypeError`: the default is `mode="sim"`, and a simulation would ignore `port=`. Add `mode="real"` |
| `Robot("so101", mode="real")` with no servo bus found | `ValueError` naming the ports on this host and that none looks like a servo bus |
| no `port=`, or `port=""`, on either driver | `ValueError` at construction naming this host's serial devices; the same sentence on `driver="lerobot"` and `driver="strands"` |
| `Robot("so101_leader", mode="real", port=...)` | `ValueError`: a leader is a `Teleoperator`, not a robot. Driving it would servo the arm a human is holding |
| `cameras=` with `driver="strands"` | `ValueError`: the native Feetech driver does not open cameras; use `driver="lerobot"` |
| a keyword the driver does not declare (`prot=`) | `ValueError` listing what `FeetechDriver` accepts |
| a port that cannot be opened | `connect_eagerly()` returns the OS error as a string; `is_connected` stays `False` |

## The operator gate

When the lerobot-driver `Robot` is mounted as an agent tool, its `execute` and `start` actions stop for operator approval before any rollout is dispatched, and refuse when no operator can be reached. The native driver's `move_to` action does not ask at this commit. [First agent](first-agent.md) shows the gate; [Drivers](../learn/hardware/drivers.md) covers the other {{n:native_drivers}} native drivers.
