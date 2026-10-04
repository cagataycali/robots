---
description: A Booster T1 takes upper-body joint targets and locomotion twists; legs never get stiffness.
---

# Booster T1

A Booster Robotics T1 humanoid takes upper-body joint targets and locomotion twists from `Robot("booster_t1", mode="real")`, and the driver never puts stiffness on a leg.

This needs the T1 on the network and the vendor SDK wheel, which is not a strands-robots extra:

```bash
pip install booster_robotics_sdk_python-*.whl      # from Booster; a pybind11 wrapper over DDS
```

```python title="sketch"
from strands_robots import Robot

t1 = Robot("booster_t1", mode="real", port="192.168.10.102", domain_id=0)
print(t1.connect_eagerly())                      # None; refuses with the install line if the SDK is absent
t1.enable_upper_body(True)                       # B1LocoClient.UpperBodyCustomControl(True): the precondition
t1.send_action({"left_shoulder_pitch": 0.3, "left_elbow_pitch": -0.5})   # radians, upper body only
t1.move(vx=0.2, vy=0.0, vyaw=0.0)                # the onboard controller walks
t1.rotate_head(pitch=0.1, yaw=0.0)
```

## The split

The T1 keeps its own balance. The legs, waist and head belong to an onboard whole-body controller; only the eight upper-body joints can be handed to a host, and only after `UpperBodyCustomControl(True)`. A `LowCmd` frame that puts gain on a leg fights the balance controller with the robot's mass behind it, and the publish reports success either way. So the driver:

| rule | how |
|---|---|
| refuses `send_action` until upper-body control is on | `enable_upper_body()` names the call |
| accepts the eight upper-body slots only | slots 2 to 9 (`UPPER_BODY_SLOTS`); head and legs name `rotate_head()` and `move()` instead |
| emits `q=0, kp=0, kd=0` for every other slot | `build_frame` |
| holds an uncommanded upper-body joint at its last observed position | commanding one arm does not drop the other |
| refuses while the fall state is not `IS_READY` | `FALL_STATE_NAMES` |
| refuses until a `LowState` frame has arrived | frame width and hold positions both come from it |

The wire literals (`mode = 0x0A` position mode, `kp=60 / kd=3`) are transcribed from the vendor's reference client shipped inside the wheel.

## Joints

| slot | name | writable |
|---|---|---|
| 0, 1 | `head_yaw`, `head_pitch` | via `rotate_head()` |
| 2 to 5 | `left_shoulder_pitch`, `left_shoulder_roll`, `left_elbow_pitch`, `left_elbow_yaw` | `send_action` |
| 6 to 9 | `right_shoulder_pitch`, `right_shoulder_roll`, `right_elbow_pitch`, `right_elbow_yaw` | `send_action` |
| 10 | `waist` | onboard controller |
| 11 to 22 | hips, knees, cranks | onboard controller, via `move()` |

The map is a module constant, so a typo in an action dict is refused even without the SDK.

## Constructor

`port=` is the robot's IP, `domain_id` the DDS domain (default 0), `cmd_type` `"parallel"` or `"serial"` as the SDK names its two command modes.

## Agent

```python title="sketch"
from strands import Agent

agent = Agent(tools=[t1])
agent("Wave with the right arm, then walk two steps forward.")
```

`execute` and `start` on the tool pause for approval; `status` and `stop` do not ([the operator gate](../agents.md#the-operator-gate)). `stop()` runs `stop_task()`: a zero twist, then upper-body control released to the onboard controller. A half that did not complete is logged as an error: a refused twist leaves the T1 walking.

## Simulation

`Robot("booster_t1")` builds the MuJoCo twin with the same joint names. The onboard balance controller has no counterpart there: a sim rollout that moves the legs exercises physics, not the robot's controller.

<robot-viewer name="booster_t1"></robot-viewer>
