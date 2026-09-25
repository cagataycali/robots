---
description: Booster T1 bring-up over booster_robotics_sdk_python - the split between the onboard whole-body controller and the eight upper-body joints a host may command.
---

# Booster T1 over the vendor SDK

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
as the G1's `unitree_sdk2py` ([recipe](unitree-g1.md#installing-the-unitree-sdk)). Without it the driver still imports, builds and
answers `get_status`; `connect_eagerly()` returns a reason naming the module and
the install line. Because the wheel is pinned to the robot's firmware
rather than resolved by this project, the *installed* build's vocabulary is an
input: `ROBOT_MODES` and the two `cmd_type` conventions are names this driver
spells so a caller sees them without the SDK, and a build that declares a
different set is refused by name — naming the enum, the member and the modes
that build does have — rather than raising out of the verb.

## See also

- [Native drivers](native-drivers.md) - the contract this driver satisfies.
- [Humanoids](../robots/humanoids.md) - the catalog entry, and the family's
  other native-driver bring-ups.
- [Unitree G1](unitree-g1.md) - the other vendor-wheel bring-up in this family.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
