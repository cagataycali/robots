---
description: The same Robot with a Driver for a body. Feetech STS3215, Unitree DDS, Microduck robotd. A real body never moves because of a default.
---

# Hardware

{{drawing:d04_twin}}

```python title="needs the arm on USB"
from strands_robots import Robot

real = Robot("so101", port="/dev/ttyACM0", confirm=lambda action: input(f"move to {action}? ") == "y")
real.calibrate(interactive=False)            # the LeRobot calibration JSON lerobot-calibrate wrote, unchanged
obs = real.observe()                         # measured joints in radians, also pushed into real.twin
real.confirm_act(obs.joints)                 # a hold, through the operator gate; the twin mirrors it
real.twin.sim.render()                       # what the arm is doing, drawn in MuJoCo
real.close()                                 # torque off, port closed, twin removed
```

| driver | hardware | in 1.0 |
|---|---|---|
| `feetech` | STS3215 buses: SO-100 / SO-101 (`pip install pyserial`) | read, write, torque, `calibrate()` |
| `unitree` | DDS `rt/lowstate`: G1 / Go2 / H1 (`unitree_sdk2py`) | observe only |
| `microduck` | robotd's JSON-RPC socket (`port=` the socket path) | observe only: robotd has no per-joint write |

A `Driver` is `names read() write() pose() images() extras() hold() estop() calibrate() close()` plus
`supports_position_hold`, `calibration` and `resolution` (None when the bus has none); `driver=` takes a name
or an instance, and an instance whose `names` are not the description's `action_keys`, in order, is refused. `Robot(name, port=)` picks the driver from the registry entry and adds a twin of the same
description to `Simulation.default()`: `real.twin` is `so101_twin`, its `twin_of` is `real`, and only
`real.is_real` is true (`twin=False` to skip). `observe()` pushes the measured joints into
the twin; `act()` mirrors the targets and steps the twin's world one control step (`manual_time()` stops it). `set_pose` raises on hardware; `real.twin.set_pose` works.

{{drawing:d07_twin_steps}}

**Calibration.** `calibrate(interactive=False)` returns the LeRobot file it found under
`$HF_LEROBOT_CALIBRATION/robots/`: `so_follower/<id>.json` (LeRobot 0.6 names every SO-10x arm so),
then `<lerobot_type>/<id>.json`; without an id, `<lerobot_type>.json` or the only file in the directory.
`Robot("so101", port=..., calibration="my_arm.json")` takes any path; a missing file is an error that
lists where it looked. `calibrate()` walks a human at a terminal through LeRobot's procedure (middle
pose, then a sweep) and writes the file; with no terminal it refuses before torque goes off. Without a
file, joints map the full turn of counts (2048 is 0 rad). `observe()` maps each joint's calibrated
`range_min`..`range_max` onto the description's joint limits, so radians match the twin; LeRobot's own
degrees are 360/4095 per count from the middle of that range, so the same pose can read a few degrees apart. A LeRobot policy
gets LeRobot's degrees: `run()` converts through the bus calibration.

**On your arm.** `STRANDS_SO101_PORT=/dev/ttyACM0 pytest tests_integ/acceptance` runs acceptance check 9
on a physical SO-101: `observe()`, a refused `confirm_act()` that writes nothing, an allowed hold the twin
mirrors, and `calibrate(interactive=False)` against LeRobot's file, and `Driver.read()` under 3 ms. Without an arm it skips and says why.

**The wire.** `FeetechBus` reads each status frame's head, `FF FF ID LEN`, then exactly `LEN` more
bytes, so `ping`, `read` and `sync_read` cost the reply (six servos answer in about a millisecond), not the
20 ms `timeout`, which only bounds a servo that does not answer. An adapter that echoes the request and
a stray byte ahead of the header are both skipped; an echo and then silence is a servo that did not answer.

{{drawing:d06_bus_frame}}

{{drawing:d05_bench}}

{{drawing:d10_what_never_reaches_the_arm}}

**The operator gate.** `act()` in your own code is the developer's call. `confirm_act()` is what an
agent tool or a dashboard calls: on a real body it asks `confirm(action)` and refuses on `False`; with
no `confirm=` it raises and says `pass confirm=`. Every refusal (the gate, a limit, the e-stop's `EStopped`) is `from strands_robots import RequestRefused`, a `PermissionError`, and writes nothing to the bus. A NaN or infinite target raises `NonFiniteTarget` (a `ValueError`) before
any write, and ends a `run()` with `stopped_by="fault"`. Only a real body
takes `confirm=`:

```python
from strands_robots import Robot

try:
    Robot("so101", confirm=lambda action: False)
except TypeError as e:
    print(e)                                 # confirm= is the operator gate of a hardware Robot; a simulated body is never gated
```
