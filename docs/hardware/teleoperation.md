---
description: Teleoperation - drive any robot or simulation from one or more LeRobot teleoperators via the Teleoperator() factory and the attach_teleop()/teleoperate() mixin.
---

# Teleoperation

Drive any `Robot` (real **or** simulated) from one or more LeRobot
teleoperators - leader arms, gamepads, keyboards, phones - through a single
high-level API.

Two pieces:

- **`Teleoperator(name, **kwargs)`** - a factory that mirrors the
  [`Robot()`](../getting-started/robot-factory.md) factory, exposing every
  teleoperator registered with LeRobot.
- **`attach_teleop()` / `teleoperate()`** - mixin methods present on every
  hardware `Robot` and every `Simulation` host. They poll each attached
  device's `get_action()`, optionally remap it, merge the results, and apply
  the merged action via the host's `send_action()`.

```python
from strands_robots import Robot, Teleoperator

follower = Robot("so101", mode="real", port="/dev/ttyACM0")
follower.attach_teleop("so101_leader", port="/dev/ttyACM1", id="leader")
follower.teleoperate()          # Ctrl+C or stop_teleoperate() to stop
```

## The `Teleoperator()` factory

```python
from strands_robots import Teleoperator

leader = Teleoperator("so101_leader", port="/dev/ttyACM1", id="leader")
```

`name` is any LeRobot-registered teleoperator type. `**kwargs` are forwarded to
that teleoperator's config (`port`, `id`, `left_port`, …) and validated -
unknown kwargs raise immediately so typos surface fast.

### Available teleoperators

| Teleoperator | Emits (action keys) |
|--------------|---------------------|
| `so100_leader`, `so101_leader` | `{motor}.pos` |
| `koch_leader`, `omx_leader`, `openarm_leader`, `openarm_mini` | `{motor}.pos` |
| `bi_so_leader`, `bi_openarm_leader` | `{motor}.pos` (dual-arm) |
| `keyboard` | joint deltas |
| `keyboard_ee` | end-effector deltas |
| `keyboard_rover` | `{linear_velocity, angular_velocity}` (WASD) |
| `gamepad` | base/EE velocities |
| `phone` | pose / EE stream |
| `homunculus_arm`, `homunculus_glove` | hand/arm joints |
| `reachy2_teleoperator` | Reachy2 joints |
| `unitree_g1` | humanoid joints |

Run `Teleoperator` against the live registry to confirm what your LeRobot
install ships:

```python
from lerobot.teleoperators.config import TeleoperatorConfig
from strands_robots.teleoperator import _ensure_lerobot_teleoperators_registered
_ensure_lerobot_teleoperators_registered()
print(sorted(TeleoperatorConfig.get_known_choices()))
```

## Mixin API

Every hardware `Robot` and `Simulation` host exposes:

| Method | What |
|--------|------|
| `attach_teleop(device_or_spec, *, name=None, method=None, map_fn=None, **kwargs)` | Register an input stream (lazy - no hardware touched). `device_or_spec` is a built teleop instance **or** a type string built via `Teleoperator(**kwargs)`. |
| `teleoperate(*, names=None, robot_name=None, hz=50.0, publish=False, block=False, duration=None)` | Run the control loop. |
| `detach_teleop(name=None)` | Remove one (or all) attached streams. |
| `stop_teleoperate()` | Stop the loop, any mesh publishers, and disconnect devices. |

### `attach_teleop`

- **`name`** - stable key for this stream (used in `teleoperate(names=[...])`,
  mesh topics, `detach_teleop`). Defaults to the device's `id`, else type.
- **`method`** - input-method label (`"arm"`, `"gamepad"`, `"keyboard"`,
  `"phone"`); auto-derived from the type when omitted.
- **`map_fn`** - optional `(action: dict) -> dict` applied **before**
  `send_action`. The bridge for cross-vocabulary teleop (e.g. EE deltas →
  joint `.pos`, or leader joint names → sim actuator names). Identity by
  default.

### `teleoperate`

- **`names`** - subset of attached streams to run (default: all).
- **`robot_name`** - target robot in a multi-robot simulation world.
- **`hz`** - control-loop rate (default `50.0`).
- **`publish`** - also publish each device to the mesh via the host's
  `start_teleop_publish` so remote peers can follow. Requires a hardware
  `Robot` host.
- **`block`** - run inline until `duration` elapses / Ctrl+C (`True`) vs
  background thread (`False`, default).
- **`duration`** - auto-stop after N seconds (`None` = until stopped).

Each tick: poll every selected device's `get_action()` → apply its `map_fn` →
**merge** (last-wins on key conflict, with a one-time warning) → apply via
`self.send_action(merged, robot_name=...)`.

## Action-key compatibility

A pairing is **zero-config** only when the teleop's action keys match what the
robot's `send_action` consumes:

| Teleop | Robot | Keys | Config |
|--------|-------|------|--------|
| `so101_leader` | `so101_follower` | `{motor}.pos` | identity ✅ |
| `keyboard_rover` | `earthrover_mini_plus` | `linear_velocity`, `angular_velocity` | identity ✅ |
| `gamepad` | `lekiwi` | base velocities | identity ✅ |
| `keyboard_ee` | `so101` (joint) | EE deltas → `.pos` | needs `map_fn` ⚠️ |
| `so101_leader` | `earthrover` | `.pos` → velocity | needs `map_fn` ⚠️ |

The merge does **not** auto-convert `.pos` ↔ `velocity`. Cross-vocabulary
pairings supply a `map_fn` - that hook exists exactly for this.

> For a wheeled rover use **`keyboard_rover`** (WASD → velocity).
> Plain `keyboard` / `keyboard_ee` emit joint / EE deltas, not base velocities.

## Recipes

### Leader arm → follower arm

```python
from strands_robots import Robot

follower = Robot("so101", mode="real", port="/dev/ttyACM0")
follower.attach_teleop("so101_leader", port="/dev/ttyACM1", id="leader")
follower.teleoperate()
```

### Earth Rover Mini+ with WASD keys

```python
rover = Robot("earthrover_mini_plus", mode="real", robot_ip="192.168.1.151")
rover.attach_teleop("keyboard_rover")              # W/A/S/D
rover.teleoperate(block=True, duration=30)         # drive 30 s, then teardown
```

### Gamepad / phone → mobile base

```python
base = Robot("lekiwi", mode="real", robot_ip="192.168.1.42")
base.attach_teleop("gamepad")                      # or "phone"
base.teleoperate()
```

### Pre-built teleop instance + explicit method

```python
from strands_robots import Robot, Teleoperator

leader = Teleoperator("koch_leader", port="/dev/ttyACM1")
follower = Robot("koch", mode="real", port="/dev/ttyACM0")
follower.attach_teleop(leader, name="leader", method="arm")
follower.teleoperate()
```

### Cross-vocabulary via `map_fn`

```python
def ee_to_joints(action: dict) -> dict:
    return my_ik(action)        # {dx,dy,dz,dgrip} -> {shoulder.pos, ...}

robot = Robot("so101", mode="real", port="/dev/ttyACM0")
robot.attach_teleop("keyboard_ee", map_fn=ee_to_joints)
robot.teleoperate()
```

### Multi-device teleop (merge inputs)

```python
robot.attach_teleop("so101_leader", port="/dev/ttyACM1", name="arm")
robot.attach_teleop("gamepad", name="base")        # different key namespace
robot.teleoperate(names=["arm", "base"])           # both stream into send_action
```

### Bimanual leader → follower

```python
bi = Robot("bi_so", mode="real", left_port="/dev/ttyACM0", right_port="/dev/ttyACM1")
bi.attach_teleop("bi_so_leader", left_port="/dev/ttyACM2", right_port="/dev/ttyACM3")
bi.teleoperate()
```

### Teleoperate a simulation (MuJoCo)

```python
from strands_robots import Simulation

sim = Simulation(...)
sim.attach_teleop(
    "so101_leader",
    port="/dev/ttyACM1",
    map_fn=lambda a: {f"sim/{k}": v for k, v in a.items()},
    robot_name="arm0",
)
sim.teleoperate(robot_name="arm0")
```

### Teleop + mesh publish (remote followers mirror)

```python
leader_host.attach_teleop("so101_leader", port="/dev/ttyACM1")
leader_host.teleoperate(publish=True)   # local drive + publish over the mesh
```

The actuation stream rides the documented [`Mesh.publish()`](../mesh.md)
chokepoint via `start_teleop_publish`. Remote followers consume it with
`start_teleop_receive` (see [Mesh teleop](robot-control.md#mesh-teleop)).

### Time-boxed / clean teardown

```python
robot.attach_teleop("so101_leader", port="/dev/ttyACM1")
robot.teleoperate(block=True, duration=60)   # 60 s then stop + disconnect
# non-block mode:
robot.teleoperate()
...
robot.stop_teleoperate()                     # stop loop + publishers + disconnect
```

## How it relates to mesh teleop

`teleoperate()` is the **local** driver: read teleop → apply to the host.
[Mesh teleop](robot-control.md#mesh-teleop) (`start_teleop_publish` /
`start_teleop_receive`) is the **transport** for streaming actions between
peers. `teleoperate(publish=True)` composes the two: drive locally **and**
publish so remote followers mirror.

## See also

- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
- [Robot control](robot-control.md) - hardware lifecycle + mesh teleop primitives.
- [Hardware tools](tools.md) - `lerobot_teleoperate` @tool for agent-driven sessions.
- [Mesh networking](../mesh.md) - the transport layer.
