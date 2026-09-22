---
description: Spawn a robot at its keyframe pose, and the solver options its MJCF carries onto the scene.
---

# Spawn pose and physics options

## Spawn pose (keyframes)

By default a robot spawns at the all-zero joint configuration. Many MuJoCo
Menagerie models ship a canonical ready pose in a MJCF `<keyframe>` (panda,
ur5e, fr3, kuka `home`; aloha `neutral_pose`; quadrupeds/humanoids a standing
`home`). Pass `keyframe=` to spawn in that pose instead - important when a
policy was trained from the home pose, since the zero configuration is
out-of-distribution:

```python
sim.add_robot(name="panda", data_config="panda", keyframe="home")  # or keyframe=0
```

The `Robot(...)` factory forwards `keyframe=` (and `orientation=`) to
`add_robot`, so a one-line spawn reaches the same pose:

```python
robot = Robot("panda", keyframe="home")
```

The pose is applied to the robot's joints by name and is restored by `reset()`,
so a keyframe spawn is sticky across episodes. A MuJoCo `<key>` pairs that pose
with the actuator command that *holds* it, and both are applied and restored
together - so a gravity-loaded arm stays at its home configuration instead of
sagging out of it as soon as the world steps. 28 of the 31 built-in robots that
ship a `<keyframe>` declare a non-zero `ctrl` in it. The keyed command is applied
verbatim, whatever quantity each actuator reads it as (a servo setpoint, a motor
torque, a stateful actuator's activation); the keyed `qvel` is not applied, since
a robot is added at rest. An unknown keyframe name/index
is an error that lists the model's available keyframes. `keyframe=None` (the
default) keeps the zero-pose spawn. (MuJoCo backend; the Newton backend rejects
`keyframe=` as not-yet-supported.)

### `position` offsets the model's own root pose

`position` is written as the attach frame's translation, and MuJoCo *composes*
that frame with the `pos` the model's root body declares - it does not replace
it. A ground-bolted arm declares `pos="0 0 0"`, so for those the offset is the
world position. A locomotion model is authored standing, so it is not:

```python
sim.add_robot(name="dog", data_config="unitree_go2", position=[0.0, 0.0, 0.4])
# Position: [0.0, 0.0, 0.845] (position=[0.0, 0.0, 0.4] + model root offset [0.0, 0.0, 0.445])
```

30 of the 55 single-root robots in the built-in registry declare a non-zero root
`pos` - the Unitree Go2 base at `z=0.445`, the JVRC pelvis at `z=1.4` - so for
those `position=[0, 0, 0]` spawns the robot standing rather than sunk into the
floor, which is the reason the compose is the useful default. `add_robot`
reports the *measured* world position of the robot's root body and names the
request and the model's offset beside it whenever they differ, so a spawn that
did not land where it was asked is visible in the result. `list_robots` reports
the same measured base pose, re-read from the physics on every call, so a robot
that has since walked, driven or fallen is listed where it now is rather than
where it spawned. This differs from `add_object`, whose `position` places its
body at exactly that world point.

### Adding a robot does not disturb the scene it joins

Only the robot being added is placed at a defined configuration - its keyframe,
or the zero pose. Everything already in the world is left exactly as it was: an
arm keeps the pose it is in (whether that is its keyframe pose or wherever a
policy or `send_action` has driven it) *and* the actuator setpoints holding it
there, objects stay where they settled or were carried to, latched `apply_force`
wrenches persist, and the clock keeps counting.

So a scene can be composed in any order, and a robot can be added mid-session
without invalidating what has already happened in it:

```python
sim.run_policy(robot_name="panda", ...)   # arm ends up somewhere useful
sim.add_robot(name="helper", data_config="so101", position=[0.0, -0.6, 0.0])
# 'panda' is still where the rollout left it; 'helper' starts at its zero pose
```

To return the *whole* world to its initial state - every robot, every object and
the clock - call `reset()`, which is what that method is for.

## Declared physics options

A robot MJCF may declare the solver settings its contacts and actuators were
tuned for. `add_robot` carries them onto the scene, because MuJoCo's `<option>`
is model-global and does not survive the spec attach:

```python
sim.create_world()
sim.add_robot(name="panda")          # model declares integrator="implicitfast"
sim.mj_model.opt.integrator          # -> mjINT_IMPLICITFAST
```

This matters for manipulation. Under the default Euler integrator a Panda's
position servos diverge enough that a top-down grasp pushes the object away and
squeezes through it; `so100`, `so101`, `aloha`, `shadow_hand` and `robotiq_2f85`
likewise declare `cone="elliptic" impratio="10"` so their grippers can hold load.

Precedence, highest first:

| Source | Wins for |
| --- | --- |
| `create_world(timestep=, gravity=)` | `timestep`, `gravity` - always |
| Your own scene MJCF (`replace_scene_mjcf`) | any field it sets |
| First robot attached that declares the field | everything else |

A model-global field holds one value, so if a second robot declares a different
value for a field already set, the existing value is kept and the discarded
request is logged with the field, both values and the robot name. Add that robot
first, or declare the value in your own scene MJCF, to make it win.

Vector environment fields (`wind`, `magnetic`, contact overrides) and the flag
bitfields describe the world rather than the robot and are never adopted.

Adoption is committed only once the robot is actually in the scene, so an
`add_robot` that reports an error leaves the world's solver settings exactly as
they were - and leaves the field free for the next robot that declares it.

## See also

- [World building](world-building.md)
- [Rough terrain](terrain.md)
