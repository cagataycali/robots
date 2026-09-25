---
description: SO-100, SO-101 and LeKiwi bring-up over the Feetech bus - reading an arm without moving it, the calibration its degrees are measured against, a threaded rollout, and the same verbs against the twin.
---

# SO arms over the Feetech bus

`so100`, `so101` and `lekiwi` reach real hardware through the
[native Feetech driver](native-drivers.md) as well as through lerobot, and the same
verbs answer from the arm's own model when `transport="twin"` is passed.

## Reading a real arm before moving it

A real arm handed to an agent can be *looked at* without running a policy on
it. `Robot(name, mode="real", port=...)` offers four observe actions that write
nothing to the servos and never ask the operator:

| action | what comes back |
|---|---|
| `get_state` (alias `get_robot_state`) | every joint: degrees, raw encoder ticks, torque on/off, supply voltage; whether the arm is calibrated |
| `list_cameras` | the cameras passed as `cameras=` and whether each is open |
| `render` | one PNG from a camera (`camera_name`, optional `output_path` inside `~/.strands_robots/renders`) |

The `render` PNG holds the colour the camera saw: a camera configured
`color_mode="bgr"` delivers frames in OpenCV's own channel order, and they are
encoded in it rather than converted a second time.

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/ttyACM0",
            cameras={"front": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}})
agent = Agent(tools=[arm])
agent("Where is the arm right now, and is torque on? Do not move it.")   # get_state, no approval
```

`get_state` opens the motor bus on first use (the bus only - the servo
configuration a rollout writes is not touched) and reads it under the same lock
a rollout and the mesh probes use. An arm with **no calibration file** is still
readable: the degrees are then an estimate from the encoder centre
(`2048 ticks = 0°`, `4096 ticks/rev`) and the text says so, naming
`lerobot-calibrate` - which is also why `execute`/`start` refuse on that arm
until it has run. A reading the arm did *not* give is reported as unread rather
than as its opposite: a calibration flag whose own read fails (on a lerobot bus
that read sweeps every servo) comes back `null` with the reason instead of
`false`, and the degrees stay whatever the arm could normalise; a
`Torque_Enable` register no motor answered is `null`, not `off`, because "torque
off" reads as "safe to move by hand". The motion actions (`execute`, `start`)
stop for operator approval; see
[security](../security/hardware.md#ros-2-dds-bridge-command-surface).

## Calibrating a Feetech SO arm

`so100`, `so101` and `lekiwi` read and command **degrees**, and those degrees are
measured against the travel `lerobot-calibrate` recorded for *that particular
arm*. Pass the file that run wrote:

```python
from strands_robots.drivers.feetech import FeetechDriver, lerobot_calibration_path

arm = FeetechDriver(
    tool_name="so101",
    port="/dev/ttyACM0",
    calibration=lerobot_calibration_path("so101_follower", "my_arm"),
)
arm.connect_eagerly()                       # returns None, or a reason
arm.send_action({"shoulder_pan": 30.0, "gripper": 100.0})
```

`lerobot_calibration_path(robot_type, robot_id)` is where
`lerobot-calibrate --robot.type=so101_follower --robot.id=my_arm` put its output,
read from LeRobot's own constants so `HF_LEROBOT_CALIBRATION` is honoured.
Records can also be passed directly (`calibration=load_calibration(path)`), and
`get_status()` reports which travel is in force as `calibration_source`.

Omitting it spans the *servo's* full rotation instead of the arm's measured
travel. No two SO-101s stop in the same place, so `0 degrees` and
`0 percent closed` then land somewhere different on each one - the degrees are an
encoder angle rather than a joint angle. Calibrate the arm and pass the file.

## Rolling a policy out on an SO arm

`so100`, `so101` and `lekiwi` run a policy on the native driver. The loop steps on
its own thread, so the verb returns at once and `get_task_status()` is the poll:

```python
arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0")
arm.run_policy(policy, instruction="pick up the cube", duration=60.0)  # 30 Hz default
arm.get_task_status()   # {"running": True, "steps": 412, "exit_reason": None, ...}
arm.stop_task()         # {"stopped": True, "steps": 604}
```

Each step reads the whole arm in one sync-read, hands the policy
`{"shoulder_pan.pos": degrees, ...}` - lerobot's own observation keys, so a
checkpoint trained on lerobot SO data needs no remap - and commands its answer
through `send_action`, in degrees (`gripper` is percent open). A setpoint the bus
refuses ends the rollout with *that* refusal as its `exit_reason`, readable after
the thread is gone. `control_frequency=` moves the pace off 30 Hz; a rate the
loop cannot pace is refused rather than divided into.

`stop_task()` halts the loop and leaves the arm energized where it stands -
dropping torque would drop a payload, and `stop` is the verb that de-energizes.
It reports `stopped=False` in an error envelope when the policy is blocking on a
remote call and the thread has not left the loop, rather than claiming a halt
`get_task_status()` would contradict. `start_task()` builds the policy from the
provider registry first, and refuses a provider it cannot build - before the arm
is committed. The same verbs run on `transport="twin"`, which is how a rollout is
rehearsed against the arm's model.

## The same agent, on the twin

`Robot("so101", mode="sim")` is the physics twin with the simulation tool's
verbs. `transport="twin"` is something else: the **Feetech driver**, with its
verbs and units, answering its bus from that model. An agent that learns to
`move_to` a pose, read `sensors` and `set_torque` here says exactly the same
words to the arm - one tool, two far ends.

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", driver="strands", transport="twin")   # FeetechDriver, model at the far end
arm.connect_eagerly()                                                    # builds the model; None, or a reason

Agent(tools=[arm])("read the joints, then move the gripper to 30 percent open")

arm.sim.render(width=640, height=480)                                    # the engine is one attribute away
arm.cleanup()                                                            # destroys an engine the driver built
```

`driver="strands"` is spelled because the SO arms' registry entries declare no
`hardware.driver`, so `Robot(..., mode="real")` alone builds the lerobot
driver, which has no twin. `so100` works the same way.

What the twin does with the bus: each motor is placed on the model through the
registry's `joint_labels` (the SO-101 asset names its joints `1`..`6`, the
SO-100's `Rotation`..`Jaw`, the bus speaks `shoulder_pan`..`gripper`) and the
actuator driving that joint; a target's degrees go through the bus's own
`to_counts` against **this arm's calibration** and the calibrated
`[range_min, range_max]` counts map linearly onto the joint's travel in the
model, so a calibration file written for a real arm places the twin where it
places the arm, and with no calibration `0 degrees` is the middle of the model's
travel. `gripper` percent spans the jaw end to end, `0` at the closed stop. A
write steps the model for one bus read period (the driver's `timeout`), so the
servo has arrived by the next `sensors`; `set_torque(False)` zeroes the
actuators' gains and the arm falls under the model's gravity, `set_torque(True)`
holds where it is, and a `move_to` while released is refused. `sensors` and the
mesh's joint reader read the model back through the same map, in degrees.
`sim=` hands in an engine you already built (with objects, a camera); `realtime=True`
steps at wall-clock speed for a viewer. The operator gate is not consulted - the
driver does not consult it on the serial bus either.

Two fidelity notes, the model's rather than the driver's: a MuJoCo position
servo settles where its gain balances the joint's friction (`frictionloss / kp`),
so a target is reached to within a degree rather than an encoder count - reads
are still reproducible to one count; and the SO-101 asset's actuators declare
`ctrlrange="0 0"` (unlimited), so a calibrated target past the joint's stops is
clamped to the joint `range` by the twin and **reported** on the reply, where
the SO-100's declared `ctrlrange` would have clamped it silently.

## See also

- [Twin transport](twin-transport.md) - which buses have a twin, and how the model stands in.
- [Teleoperation](teleoperation.md) - driving a follower from a leader arm.
- [Arms](../robots/arms.md) - the catalog entry and the compatibility table.
