---
description: Compose non-trivial scenes - multiple robots, tables, obstacles, custom MJCF.
---

# World building

```python
from strands_robots import Robot

sim = Robot("so100")                              # one arm on flat ground plane
sim.add_robot(name="arm2", data_config="so100", position=[0.0, 0.5, 0.0])   # second arm

sim.add_object(name="table", shape="box", size=[0.5, 0.5, 0.02],
               position=[0.0, 0.0, 0.0], color=[0.5, 0.3, 0.1, 1.0], mass=20.0)

sim.add_camera(name="overhead", position=[0.0, 0.0, 1.5], target=[0.0, 0.0, 0.0])
```

## Setup entry points

`Robot("so100")` is the one-step way to get a ready-to-drive engine: it builds
the world and adds the named robot for you. Constructing a backend directly -
`create_simulation("mujoco")` or `Simulation()` - gives an **empty** engine; you
then call `create_world()` and `add_robot("so100")` yourself.

Because `Robot(...)` has already built the world, calling `create_world()` on
what it returns is refused - a world cannot be rebuilt under a live scene. The
refusal names what that world holds and which call applies the arguments you
passed:

| You asked for | What applies it |
|---------------|-----------------|
| `timestep=`, `gravity=` | `set_timestep` / `set_gravity` on the live world - contents kept |
| `ground_plane=`, `terrain=`, `difficulty=` | compiled in at creation: `destroy()`, then `create_world(...)` |
| nothing | the world is ready; `reset()` restarts the rollout in place |

`reset()` applies no `create_world` parameter - it restores the initial state at
the values the world was built with - so it is never the way to get a *different*
world.

`robot_name` therefore belongs to `Robot(...)` and `add_robot(...)`, never to a
backend constructor. Passing it to the constructor
(`Simulation(robot_name="so100")`) is rejected with a `TypeError` rather than
silently ignored, so the mistake is caught up front instead of surfacing later
as an unrelated `No world` error.

### Unrecognised constructor keywords

A backend constructor's `**kwargs` is a *tolerating* sink: a name it cannot bind
is dropped, which is what lets one call carry another backend's options and
resolve against whichever backend is selected.

| You passed | Outcome |
|------------|---------|
| a name this backend binds (`default_timestep=0.001`) | applied |
| a name no backend binds, but close to one this backend binds (`defualt_timestep=0.001`) | **`TypeError`** naming `default_timestep` |
| a name another backend binds (`num_envs=4`, `device="cuda"`, a plugin's `timestep=`) | tolerated, dropped, logged at DEBUG |

The middle row is the one that used to be silent, and it is the reason the sink
is not simply permissive: dropping a misspelling made it byte-identical to
omitting the argument, so `Robot("so101", defualt_timestep=0.001)` integrated the
physics at the 2 ms default -- half the requested rate -- and reported success.
No portable call can intend a misspelling of a parameter the receiver itself
reads, so exactly that subset is refused while the portability case above it is
untouched.

`Robot(name, mode="sim")` screens its own parameters the same way, since it
forwards the rest verbatim: `Robot("so101", positon=[0.5, 0, 0])` named
`position` instead of spawning the robot at the origin.

## Strategies

| Need | Approach |
|------|----------|
| Add robots / objects incrementally | `add_robot` / `add_object` / `add_camera` ([objects](objects.md)) |
| Replace entire world | `load_scene(scene_path=...)` |
| Procedural scene | loop over `add_object` ([objects](objects.md)) |
| Raw MJCF tweak without recompile | `patch_scene_mjcf(ops)` ([scene editing](scene-editing.md)) |

## Cameras

Free cameras look from `position` toward `target` (`fov=60.0`, `width=640`, `height=480`). Robot-URDF cameras (wrist, etc.) are auto-discovered on `add_robot` - no `add_camera` needed.

A discovered camera is registered under its short MJCF name (`wrist`), and the
compiled model also carries it namespaced (`so101/wrist`). Either spelling
addresses it on every camera surface - `render`, `render_depth`, `get_frame`,
`get_camera_params` and the recorders - the same way a body name may be bare or
namespaced; `get_observation` keys its frame on the short one, and
`list_cameras` offers both. The short name is
first-come across robots: when a second robot declares a camera whose short name
is already taken, that camera is registered under its namespaced name instead
(logged, naming both), so two arms that both declare `wrist` give you `wrist` and
`arm2/wrist` rather than one of them shadowing the other. Each camera belongs to
exactly one robot, which is what makes `remove_robot` take that robot's cameras
with it and leave every other robot's alone.

To mount a camera ON a moving body (a realistic wrist/gripper view that rides with the arm), pass `parent_body`. Body names are namespaced `<robot>/<body>`; discover the exact mount point with `list_bodies` instead of guessing:

```python
bodies = sim.list_bodies(robot_name="so101")["content"][1]["json"]
mount = bodies["gripper_body"]          # e.g. "so101/gripper" -- the wrist mount
sim.add_camera(name="wrist", parent_body=mount,
               position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1])  # local frame
```

`list_bodies()` (no `robot_name`) lists every body in the world; with `robot_name` it scopes to that robot and also returns `gripper_body`, the best-guess end-effector mount.

The guess matches its hint words (`gripper`, `hand`, `jaw`, `ee`, `tool` - one set, read by every backend) on word boundaries, so a short hint cannot fire inside an unrelated word - a `knee` link or a `wheel` hub is not a gripper mount because `ee` occurs in its name. A robot with no gripper-like body reports `gripper_body: None` and omits the mount line rather than naming an unrelated body; pick the mount from the full `bodies` list in that case. `jaw` is in the set because the SO-100 family names its gripper bodies `Fixed_Jaw` / `Moving_Jaw`, so the mount resolves for those arms too.

A mounted camera survives `remove_robot`, which rebuilds the whole scene: it is
re-mounted on its body once every surviving robot is re-attached, keeping its
local pose and its tracking. Removing the robot the camera is mounted ON leaves
it with no mount point, so that camera is dropped (with a warning naming it)
rather than blocking the removal.

That rebuild is faithful to the registry, and only to the registry, which is why
`remove_robot` is refused on a world built by `load_scene`. A loaded scene's
bodies, lights, tendons and equality constraints live only in the compiled spec,
so rebuilding from `robots` / `objects` / `cameras` would drop all of them; the
refusal comes before anything is touched, so the scene is left exactly as it was.
To get the same world without one robot, `load_scene` again and `add_robot` only
the robots you want, or swap the scene wholesale with `replace_scene_mjcf`. The
additive verbs need no such gate - `add_robot`, `add_object` and `add_camera`
mutate the loaded spec in place and preserve it.

## Multi-robot policies

```python
from strands_robots.policies import create_policy

sim.run_multi_policy(
    policies={"so100": create_policy("mock"), "panda": create_policy("mock")},
    instructions={"so100": "pick cube", "panda": "hold tray"},
    duration=10.0,
)
```

## See also

- [Spawn pose and physics options](spawning.md)
- [Rough terrain](terrain.md)
- [Objects](objects.md)
- [Meshes and materials](meshes-and-materials.md)
- [Editing and exporting a scene](scene-editing.md)
- [Simulation overview](overview.md)
- [Domain randomization](domain-randomization.md)
