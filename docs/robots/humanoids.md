---
description: Full-body humanoids and expressive desktop robots.
---

# Humanoids

Full-body humanoids and expressive desktop robots.

```python
from strands_robots import Robot
sim = Robot("unitree_g1")       # Unitree G1
sim = Robot("unitree_h1")       # Unitree H1
sim = Robot("apollo")           # Apptronik Apollo
sim = Robot("reachy_mini")      # Pollen Reachy Mini (expressive)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:humanoid, expressive}}

## Real hardware: the Booster T1 native driver

The T1 is driven natively through its own SDK
(`booster_robotics_sdk_python`, a pybind11 wrapper over the robot's DDS
transport) — lerobot has no robot type for it, so `driver="strands"` is the only
way to reach it and the registry declares it as the default:

```python
from strands_robots import Robot

t1 = Robot("booster_t1", mode="real", port="192.168.10.102")  # driver="strands"
t1.connect_eagerly()          # ChannelFactory + loco client + LowState/LowCmd channels

t1.move(vx=0.2)               # walk: a twist to the onboard controller
t1.rotate_head(pitch=0.1, yaw=-0.3)

t1.enable_upper_body(True)    # claim the arms (UpperBodyCustomControl)
t1.send_action({"left_shoulder_pitch": -0.4, "right_elbow_pitch": 0.7})
t1.stop_task()                # halt locomotion + hand the arms back
```

Control of the T1 is **split**, and the driver enforces the split rather than
documenting it. An onboard whole-body controller owns the legs, waist and head;
the eight upper-body joints can be handed to a host, and only then:

| Slots | Joints | Who commands them |
|-------|--------|-------------------|
| 2–9 | `left`/`right` `shoulder_pitch`, `shoulder_roll`, `elbow_pitch`, `elbow_yaw` | the host, via `send_action`, after `enable_upper_body(True)` |
| 0–1 | `head_yaw`, `head_pitch` | the robot, via `rotate_head()` |
| 10–22 | waist, hips, knees, ankle cranks | the robot, via `move()` |

Every frame the driver publishes puts `kp=kd=0` on every slot outside 2–9, which
is what leaves the onboard controller in charge of balance; an uncommanded arm
joint holds its last *observed* position, so commanding one arm does not drop the
other. `send_action` refuses before the gate is open, before the first `LowState`
has arrived (the frame width and the hold positions both come from the robot's
own report), and for any joint outside the upper body — naming `move()` or
`rotate_head()` instead.

The driver also subscribes the T1's battery and fall-down topics. A fall state
other than `IS_READY` refuses a write — a held arm posture is noise while the
robot is on its way to, on, or getting off the floor, and can obstruct its
getting-up routine — and the gate reads *evidence of a fall* rather than the
absence of a reading, so a T1 whose fall topic is silent keeps writing. The
charge read reaches `get_status()` as the shared `battery_pct` field and gates
nothing: the SDK names it `soc` and documents no scale, and a floor compared
against an unverified scale refuses every frame or none while looking like a
working check.

`run_policy`/`start_task` refuse: this driver publishes one frame per call and
owns no control loop. A caller who wants a trajectory calls `send_action` on
their own timer (the vendor's reference client runs 100 Hz).

The SDK is a vendor wheel rather than a declared dependency of this project
(`pip install booster_robotics_sdk_python`, linux wheels only) — the same footing
as the G1's `unitree-sdk2`. Without it the driver still imports, builds and
answers `get_status`; `connect_eagerly()` returns a reason naming the module and
the install line. Because the wheel is pinned to the robot's firmware
rather than resolved by this project, the *installed* build's vocabulary is an
input: `ROBOT_MODES` and the two `cmd_type` conventions are names this driver
spells so a caller sees them without the SDK, and a build that declares a
different set is refused by name — naming the enum, the member and the modes
that build does have — rather than raising out of the verb.

## Real hardware: the Microduck robotd driver

The Microduck is driven natively through its on-robot
`robotd` daemon (Pollen's `duck-ipc-proto` JSON-RPC over a Unix socket) — the
same policy code that runs in sim drives the physical robot:

```python
from strands_robots import Robot

# On the robot (robotd's default socket), or a socket forwarded over SSH.
duck = Robot("microduck", mode="real")                       # /run/robotd.sock
duck = Robot("microduck", mode="real", port="/tmp/robotd.sock")

duck.connect_eagerly()                 # Hello handshake + subscribe to state
duck.send_action({"vx": 0.15})         # walk forward (robot.move intent)
duck.send_action({"skill": "kick_left"})  # a named skill (robot.do)
duck.emergency_stop()                  # robot.stop
```

Every intent frame carries its whole group -- `robot.move` always carries
`vx`/`vy`/`vyaw`, `robot.pose` always `z`/`roll`/`pitch`/`active` -- so an
absent key is the resting value (`{"vx": 0.15}` walks straight ahead) and a key
this driver does not know is refused, even with a known key beside it. That
distinction matters because the two are indistinguishable on the wire:
`{"vx": 0.15, "yaw": 0.6}` -- `yaw` being the spelling the `get_status` pose
block uses for the heading -- would otherwise send `vyaw: 0`, walk straight past
the turn and report success. The same refusal catches a 14-joint
`MICRODUCK_JOINT_NAMES` action: four of its keys are head axes, so it would
arrive as a `robot.head` frame with the other ten joints dropped, which is the
per-joint stream `run_policy` refuses by name.

All three halt paths - `stop()`, `stop_task()` and `emergency_stop()` - send the
same `robot.stop`, and an accepted one is recorded in
`get_status()["motion_stopped"]`, the field an operator reads to decide whether
the robot is safe to approach. Only an accepted halt sets it: a stop robotd
declines leaves it false. `relax()` and `enable_torque(False)` de-energise rather
than halt a commanded motion, so they leave the flag alone.

`robotd` owns the walking/skill ONNX on-device, so `run_policy`/`start_task`
refuse and point back at the intent path; use `mode="sim"` for a host-driven
[`MicroduckPolicy` rollout](../policies/microduck.md#walking-in-mujoco). For a remote robot, forward its socket to a local
path (`ssh -L`/`socat`) and pass that path as `port=`.

## Mounting a camera on a humanoid

`add_camera(parent_body=...)` mounts a camera ON a body so it rides with the
robot, and `position`/`target` are then in that body's LOCAL frame. The general
recipe in [World building](../simulation/world-building.md) reads the mount from
`list_bodies(robot_name=...)["gripper_body"]`, which is the right mount for an
arm. A humanoid here reports `gripper_body: None`: that hint set (`gripper`,
`hand`, `jaw`, `ee`, `tool`) is arm-shaped, and matching it on word boundaries is
what keeps a leg out of the answer - a `knee` link is not an end-effector because
`ee` occurs in its name. Pick the mount from the full `bodies` list instead.

For a head camera there is no head link to pick. The Unitree G1 asset's body tree
ends at the wrists - `pelvis` to hips/knees/ankles, and `waist_yaw_link` to
`waist_roll_link` to `torso_link` to shoulders/elbows/wrists - with no
`head_link`, `neck_link` or `eye_link`; its only sites are two IMUs and the two
feet. Mount on the torso with a local offset instead:

```python
sim.add_camera(name="head", parent_body="g1/torso_link",
               position=[0.08, 0.0, 0.35], target=[1.0, 0.0, 0.2])
```

That puts the camera 0.35 m above the torso frame, roughly head height, and it
rides with the torso through waist yaw and roll. An arm camera mounts the same
way on a wrist link (`g1/left_wrist_yaw_link`).

## See also

- [Mobile](mobile.md) - quadrupeds and wheeled bases.
- [Bimanual](bimanual.md) - two-arm rigs without the legs.
- [GR00T](../policies/groot.md) - many GR00T data_configs target humanoids.
