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
  the merged action via the host's `send_action()`. Every keyword, refusal and
  per-tick rule is on [The teleoperation loop](teleoperation-loop.md).

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

### A leader arm is a teleoperator, not a `Robot`

A leader carries the same servo bus as the follower it drives, so its name reads
like a robot name - but `Robot()` builds a *follower* driver, which would
torque-enable the arm you are holding. `Robot()` refuses every `*_leader` name
and points here:

```python
Robot("so101_leader", mode="real", port="/dev/ttyACM1")
# ValueError: 'so101_leader' is a teleoperator (leader) device, not a robot.
#   Build it with ``Teleoperator('so101_leader', port=...)`` and attach it to
#   the follower it drives ...
```

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
from strands_robots.utils import ensure_lerobot_family_registered
ensure_lerobot_family_registered("teleoperators")
print(sorted(TeleoperatorConfig.get_known_choices()))
```

## Recipes

Every `attach_teleop` / `teleoperate` keyword these use, and whether a pairing
needs a `map_fn`, is on [The teleoperation loop](teleoperation-loop.md).

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

A bimanual device is two arms, so each side carries its own config object -
there is no single `port`. The registered follower name is `bi_so_follower`, and
both `BiSOFollowerConfig` and `BiSOLeaderConfig` require a `left_arm_config` /
`right_arm_config` pair.

```python
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

bi = Robot(
    "bi_so_follower",
    mode="real",
    left_arm_config=SOFollowerConfig(port="/dev/ttyACM0"),
    right_arm_config=SOFollowerConfig(port="/dev/ttyACM1"),
)
bi.attach_teleop(
    "bi_so_leader",
    left_arm_config=SOLeaderConfig(port="/dev/ttyACM2"),
    right_arm_config=SOLeaderConfig(port="/dev/ttyACM3"),
)
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
)
sim.teleoperate(robot_name="arm0")   # the target robot is a teleoperate() kwarg
```

### Teleop + mesh publish (remote followers mirror)

```python
leader_host.attach_teleop("so101_leader", port="/dev/ttyACM1")
leader_host.teleoperate(publish=True)   # local drive + publish over the mesh
```

The actuation stream rides the documented [`Mesh.publish()`](../mesh.md)
chokepoint via `start_teleop_publish`. Remote followers consume it with
`start_teleop_receive` (see [Mesh teleop](robot-control.md#mesh-teleop)).

Every door that accepts a teleoperator grades one contract - a callable
`get_action()` - whether the device is attached locally, handed to
`start_teleop_publish`, or used to build an `InputPublisher`. A device without it
is refused at the call rather than starting a session that publishes nothing.

### Time-boxed / clean teardown

```python
robot.attach_teleop("so101_leader", port="/dev/ttyACM1")
robot.teleoperate(block=True, duration=60)   # 60 s then stop + disconnect
# non-block mode:
robot.teleoperate()
...
robot.stop_teleoperate()                     # stop loop + publishers + disconnect
#   -> status="error" + stopped=false if the loop is still polling the leader;
#      get_teleoperate_status()["thread_alive"] reads the loop thread itself.
```

## How it relates to mesh teleop

`teleoperate()` is the **local** driver: read teleop → apply to the host.
[Mesh teleop](robot-control.md#mesh-teleop) (`start_teleop_publish` /
`start_teleop_receive`) is the **transport** for streaming actions between
peers. `teleoperate(publish=True)` composes the two: drive locally **and**
publish so remote followers mirror.

Because that composition drives both followers from one `get_action()` stream,
both paths hold a frame to a per-joint slew bound - the local loop to
`STRANDS_TELEOP_SLEW_ABS`, the mesh receive path to
`STRANDS_MESH_INPUT_SLEW_ABS` - otherwise the follower next to the operator
would be the unguarded one. The mesh receive path adds guards the local path has
no need of, since it accepts frames from another host: sender scoping, replay freshness, an
apply-rate ceiling (`STRANDS_MESH_INPUT_MAX_HZ`) and a magnitude clamp
(`STRANDS_MESH_INPUT_VALUE_ABS`).

## See also

- [The teleoperation loop](teleoperation-loop.md) - the mixin API and the slew bound.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
- [Robot control](robot-control.md) - hardware lifecycle + mesh teleop primitives.
- [Hardware tools](tools.md) - `lerobot_teleoperate` @tool for agent-driven sessions.
- [Mesh networking](../mesh.md) - the transport layer.
