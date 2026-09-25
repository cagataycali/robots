---
description: Microduck bring-up over the on-robot robotd daemon - socket discovery, the intent vocabulary, and the twist envelope refused before the wire.
---

# Microduck over the robotd link

The Microduck is driven natively through its on-robot
`robotd` daemon (Pollen's `duck-ipc-proto` JSON-RPC over a Unix socket, API
version 31 / microduck 0.14.1) — the same policy code that runs in sim drives
the physical robot. With no `port`, the driver finds the socket itself:

| Where you run | What to set | How it connects |
|---|---|---|
| On the duck | nothing | `/run/robotd.sock` |
| Another machine, key-authenticated ssh to the board | `MICRODUCK_HOST=[user@]<duck ip>` (user defaults to `radxa`, or `DUCK_BOARD_USER`) | the driver runs `ssh -N -L` and forwards robotd's, mediad's and tofd's sockets |
| You forwarded the socket yourself | `MICRODUCK_SOCKET=/path/to/local.sock` | that path |

```python
from strands_robots import Robot

duck = Robot("microduck", mode="real")            # discovers the socket (table above)
duck = Robot("microduck", mode="real", port="ssh://radxa@10.0.0.5")   # explicit forward

duck.connect_eagerly()                 # optional - the first verb connects on its own
duck.send_action({"vx": 0.15})         # walk forward (robot.move intent, one frame)
duck.send_action({"skill": "kick_left"})  # a named skill (robot.do)
duck.emergency_stop()                  # robot.stop
```

As an agent tool the driver exposes the whole operator vocabulary as one
`action`: `move` (a bounded twist kept alive past robotd's 0.5 s deadman and
ended with a zero twist), `head`, `look_at` (a point in the robot frame, solved
on the robot), `pose`, `mouth`, `do`/`skills` (the skills *this* robot lists),
`sit`/`stand`, `enable`/`disable`, `relax`/`init`/`reboot_motors` (each needs
`confirm=true`), `sounds`/`play_sound`, `theremin`, `mode`/`set_mode`
(walk/roller), `policies`/`load_policy`/`reload_policies`, `health`, `version`,
`model`, `odometry`, `monitor`, `camera` (one JPEG from mediad) and `tof` (one
depth-frame summary from tofd), beside the universal `sensors`/`status`/`stop`.
`Robot("microduck", mode="auto")` asks the driver's `probe_hardware()` first
(one Hello on the discovered socket) and is the real robot when a robotd
answers, MuJoCo when none does. A twist outside the pad's envelope (walk `|vx|,|vy| <= 0.3` m/s,
`|vyaw| <= 1.5` rad/s; roller `vx` in `[-0.5, 0.6]`, no strafe) is refused
before the wire, because robotd clamps nothing there; robotd's own
`accepted: false` comes back as the verb's refusal with its `reason`.

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
[`MicroduckPolicy` rollout](../policies/microduck.md#walking-in-mujoco).

## See also

- [Native drivers](native-drivers.md) - the contract this driver satisfies.
- [Humanoids](../robots/humanoids.md) - the catalog entry, and the family's
  other native-driver bring-ups.
- [Microduck policy](../policies/microduck.md) - the host-driven MuJoCo rollout.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
