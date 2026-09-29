---
description: A leader arm, a gamepad or any get_action() object drives a follower at a fixed rate, with a speed bound that refuses glitch frames.
---

# Teleoperation

At the end of this page a leader arm, a gamepad or any object with a `get_action()` drives a follower, in simulation or on hardware, at a fixed rate, with a per-joint speed bound that refuses glitch frames instead of clamping them, and optionally mirrored to remote followers over the mesh.

This runs without hardware: the "teleoperator" is a sine wave.

```python
import math, time
from strands_robots import Robot

class Sine:
    is_connected = False
    def connect(self): self.is_connected = True
    def disconnect(self): self.is_connected = False
    def get_action(self):
        return {"shoulder_pan.pos": 20 * math.sin(time.time())}   # a leader arm speaks lerobot keys

sim = Robot("so101")
sim.attach_teleop(Sine(), name="sine", map_fn=lambda a: {"1": math.radians(a["shoulder_pan.pos"])})
result = sim.teleoperate(hz=50, duration=1.0, block=True)
print(result["content"][1]["json"])
# {'frames': 50, 'errors': 0, 'slew_rejected': 0, 'hz_actual': 49.9, ..., 'status': 'success'}
```

## Two halves

`Teleoperator(name, **kwargs)` is the input-device factory, the sibling of `Robot`. It resolves lerobot's `TeleoperatorConfig` registry, so every teleoperator lerobot ships is available by name: `so101_leader`, `koch_leader`, `gamepad`, `keyboard`, `keyboard_ee`, `phone`, and the bimanual variants. It needs `[lerobot]`.

`TeleopMixin` is the loop, shared by the hardware `Robot` and the MuJoCo simulation, so the same code drives both:

| call | does |
|---|---|
| `attach_teleop(device, name=, map_fn=)` | registers a device; opens nothing |
| `teleoperate(hz=50, publish=False, block=False, duration=None, names=None, robot_name=None)` | connects the selected devices and runs the loop, in a background thread unless `block=True` |
| `stop_teleoperate()` | ends a background session |
| `detach_teleop(name)` | removes a device |

Each tick polls every attached device, applies its `map_fn`, merges the dicts (last wins on a key conflict, warned once) and calls `send_action(merged)`. Several devices at once are allowed: a leader arm for the joints and a gamepad for the gripper.

## Hardware

Needs a follower and a leader on two ports and `[lerobot]`:

```python title="sketch"
from strands_robots import Robot, Teleoperator

follower = Robot("so101", mode="real", port="/dev/ttyACM0")
leader = Teleoperator("so101_leader", port="/dev/ttyACM1", id="blue")
follower.attach_teleop(leader)
follower.teleoperate(block=True)          # Ctrl+C to stop
```

`map_fn` is the bridge when the two sides name joints differently, which is how a real leader drives a simulated follower: lerobot keys in, MuJoCo actuator names out, as in the first fence.

## The slew bound

Every merged frame is held to `STRANDS_TELEOP_SLEW_ABS` units per second per joint (default 500, wide enough for degree-valued arms and 0 to 100 grippers). A frame that exceeds it is refused and counted in `slew_rejected`, not clamped, because clamping toward a commanded value silently alters an actuator command. A physical leader cannot produce that speed; an encoder glitch or a USB re-enumerate can. A session with any refusal does not report `success`. The mesh receive path applies the same bound to inbound frames, so the follower next to the operator and a remote one judge a frame identically.

## Over the mesh

`teleoperate(publish=True)` also publishes each device's stream through the host's `start_teleop_publish`, so a remote peer can mirror it. That needs a live mesh (`STRANDS_MESH=true`, see [mesh](../mesh/index.md)) on both ends.

## Recording while teleoperating

The agent-facing `lerobot_teleoperate` tool wraps lerobot's own teleop and record scripts as managed sessions (`action="start"` with `dataset_repo_id=` records episodes; `status`, `stop`, `list`). For the in-process recorder that works on the sim too, see [record](../data/record.md).

## Refusals you will meet

`teleoperate` grades its arguments before connecting anything: `hz` and `duration` must be positive finite numbers, `names=[]` is refused rather than read as "all", a `robot_name` the host cannot route to is refused before any device opens. In a multi-robot simulation, name the follower with `robot_name=`.
