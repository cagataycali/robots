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

A `Driver` is `names read() write() pose() images() extras() calibrate() close()`; `driver=` takes a name
or an instance. `Robot(name, port=)` picks the driver from the registry entry and adds a twin of the same
description to `Simulation.default()`: `real.twin` is `so101_twin`, its `twin_of` is `real`, and only
`real.is_real` is true (`twin=False` to skip). `observe()` pushes the measured joints into
the twin, `act()` mirrors the targets. `set_pose` raises on hardware; `real.twin.set_pose` works.

**Calibration.** `calibrate(interactive=False)` returns the file found at
`$HF_LEROBOT_CALIBRATION/robots/<lerobot_type>/<id>.json`; `calibrate()` walks a human through
LeRobot's procedure (middle pose, then a sweep) and writes the same file. Without one, joints map the
full turn of counts (2048 is 0 rad).

**On your arm.** `STRANDS_SO101_PORT=/dev/ttyACM0 pytest tests_integ/acceptance` runs acceptance check 9
on a physical SO-101: `observe()`, a refused `confirm_act()` that writes nothing, an allowed hold the twin
mirrors, and `calibrate(interactive=False)` against LeRobot's file, and `Driver.read()` under 3 ms. Without an arm it skips and says why.

**The wire.** `FeetechBus` reads each status frame's head, `FF FF ID LEN`, then exactly `LEN` more
bytes, so `ping`, `read` and `sync_read` cost the reply (six servos answer in about a millisecond), not the
20 ms `timeout`, which only bounds a servo that does not answer. An adapter that echoes the request and
a stray byte ahead of the header are both skipped.

{{drawing:d06_bus_frame}}

{{drawing:d05_bench}}

**The operator gate.** `act()` in your own code is the developer's call. `confirm_act()` is what an
agent tool or a dashboard calls: on a real body it asks `confirm(action)` and refuses on `False`; with
no `confirm=` it raises `PermissionError` and says `pass confirm=`. A refusal writes nothing to the bus. Only a real body
takes `confirm=`:

```python
from strands_robots import Robot

try:
    Robot("so101", confirm=lambda action: False)
except TypeError as e:
    print(e)                                 # confirm= is the operator gate of a hardware Robot; a simulated body is never gated
```
