---
description: Compose non-trivial scenes - multiple robots, tables, obstacles, custom MJCF.
---

# World building

```python
from strands_robots import Robot

sim = Robot("so100")                              # one arm on flat ground plane
sim.add_robot(name="so100", position=[0.0, 0.5, 0.0])   # second arm

sim.add_object(name="table", shape="box", size=[0.5, 0.5, 0.02],
               position=[0.0, 0.0, 0.0], color=[0.5, 0.3, 0.1, 1.0], mass=20.0)

sim.add_camera(name="overhead", position=[0.0, 0.0, 1.5], target=[0.0, 0.0, 0.0])
```

## Setup entry points

`Robot("so100")` builds the world and adds the named robot. A backend
constructed directly (`create_simulation("mujoco")`, `Simulation()`) is empty:
call `create_world()` and `add_robot("so100")` yourself. `create_world()` on a
live world is refused; the refusal names which call applies what you passed:

| You asked for | What applies it |
|---------------|-----------------|
| `timestep=`, `gravity=` | `set_timestep` / `set_gravity` on the live world - contents kept |
| `ground_plane=`, `terrain=`, `difficulty=` | compiled in at creation: `destroy()`, then `create_world(...)` |
| nothing | the world is ready; `reset()` restarts the rollout in place |

`reset()` restores the state the world was built with; it never builds a
different world. `robot_name` belongs to `Robot(...)` / `add_robot(...)`, and
a constructor refuses it with a `TypeError`.

### Unrecognised constructor keywords

A backend constructor's `**kwargs` tolerates another backend's options but
refuses a misspelling of its own:

| You passed | Outcome |
|------------|---------|
| a name this backend binds (`default_timestep=0.001`) | applied |
| a name no backend binds, but close to one this backend binds (`defualt_timestep=0.001`) | **`TypeError`** naming `default_timestep` |
| a name another backend binds (`num_envs=4`, `device="cuda"`, a plugin's `timestep=`) | tolerated, dropped, logged at DEBUG |

`Robot(name, mode="sim")` screens its own parameters the same way
(`positon=` names `position`).

## Strategies

| Need | Approach |
|------|----------|
| Add robots / objects incrementally | `add_robot` / `add_object` / `add_camera` |
| Replace entire world | `load_scene(scene_path=...)` |
| Procedural scene | loop over `add_object` |
| Raw MJCF tweak without recompile | `patch_scene_mjcf(ops)` |

## Spawn pose (keyframes)

Robots spawn at the all-zero joint configuration unless `keyframe=` names a
MJCF `<keyframe>` (`"home"` on panda, ur5e, fr3, kuka and the quadrupeds;
`neutral_pose` on aloha) - important when a policy was trained from the home
pose. The keyed `qpos` and `ctrl` are applied together and restored by
`reset()`, so a gravity-loaded arm holds its pose. An unknown keyframe lists
the model's keyframes. MuJoCo only; Newton rejects `keyframe=`.

```python
sim.add_robot(name="panda", data_config="panda", keyframe="home")  # or keyframe=0
```

```python
robot = Robot("panda", keyframe="home")
```

`position` is the attach frame's translation, composed with the `pos` the
model's root body declares (a Go2 at `position=[0, 0, 0.4]` lands at
`z=0.845`); `add_robot` reports the measured root position and the model
offset whenever they differ. `add_object`'s `position` is the exact world
point. Adding a robot places only that robot; everything already in the world
- poses, setpoints, latched wrenches, the clock - is untouched.

## Declared physics options

`add_robot` carries the solver settings a robot MJCF declares (`integrator`,
`cone`, `impratio`) onto the scene, because MuJoCo's `<option>` is
model-global and does not survive the spec attach. Precedence, highest first:

| Source | Wins for |
| --- | --- |
| `create_world(timestep=, gravity=)` | `timestep`, `gravity` - always |
| Your own scene MJCF (`replace_scene_mjcf`) | any field it sets |
| First robot attached that declares the field | everything else |

A second robot declaring a different value is logged and ignored; add it first
or set the field in your own scene MJCF. Vector environment fields (`wind`,
`magnetic`) and flag bitfields are never adopted.

## Rough terrain

`create_world(terrain=...)` replaces the flat plane with a deterministic
heightfield (same +/-5 m footprint, 0 to ~8 cm, flush with `z=0`, regenerated
identically on every `reset()`). Four kinds ship: `rough` (value-noise bumps),
`stairs` (plateaus rising along +x), `pyramid` (concentric plateaus rising to
the centre) and `slope` (constant grade along +x). `difficulty=` scales the
peak height (`1.0` default, `<1` gentler, `>1` harsher; a finite number `> 0`,
refused on a flat world). MuJoCo only; Newton rejects `terrain=`.

```python
sim.create_world(terrain="rough")        # bumpy heightfield ground
sim.add_robot("unitree_go2", keyframe="home")
```

```python
sim.create_world(terrain="rough", difficulty=0.3)  # gentle bumps (early stage)
# ... later, harder stages ...
sim.create_world(terrain="rough", difficulty=1.0)  # full ~8 cm bumps (default)
sim.create_world(terrain="rough", difficulty=2.0)  # exaggerated ~16 cm bumps
```

A floating-base robot added to or reset in a terrain world is seated on the
local surface height under its `(x, y)` - its own free joint, resolved by
ownership, so a free-jointed task object shipped in the robot's MJCF is never
moved. The same base is what `get_observation` reports as `base_pos` /
`base_quat` and what `start_recording` records.

## Procedural objects

```python
import random

sim = Robot("so100")

for i in range(5):
    sim.add_object(
        name=f"cube_{i}", shape="box", size=[0.025, 0.025, 0.025],
        position=[random.uniform(0.2, 0.5), random.uniform(-0.15, 0.15), 0.025],
        color=[random.random(), random.random(), random.random(), 1.0],
    )
```

## Object shapes and size

`shape` takes `box`, `sphere`, `cylinder`, `capsule`, `ellipsoid`, `plane` or
`mesh`. `size` is the **full extent in meters** along each local axis - not
MuJoCo's half-extent - so `size=[0.05, 0.05, 0.05]` is a 5 cm cube. Pass every
component the shape consumes; a partial or empty vector is refused rather than
completed, and omitting `size` gives the 5 cm default:

| Shape | Components consumed |
|-------|---------------------|
| `box` / `ellipsoid` | `[x, y, z]` - all three full edge lengths / diameters |
| `cylinder` | `[diameter, unused, full height]` - three (index 1 is ignored) |
| `capsule` | `[diameter, unused, cylinder-section length]` - three (index 1 is ignored). The two caps add `size[0] / 2` at each end, so the object stands `size[2] + size[0]` tall |
| `sphere` | `[diameter]` - one is enough |
| `plane` | `[x]` or `[x, y]` visual half-widths (`y` mirrors `x` when omitted) |
| `mesh` | none - the asset's own units define the extent |

The success text reports the extent the geom **compiled to**, read back off
the model (`capsule` `[0.05, 0, 0.9]` stands `0.95` m tall). `set_geom_properties(size=...)`
takes MuJoCo's own `geom_size` components instead - see
[Domain randomization](domain-randomization.md). The per-shape counts are
MuJoCo's alone; Newton and Isaac accept a short `size`
([#1858](https://github.com/strands-labs/robots/issues/1858)).

`mass` (kg) applies to dynamic objects and must be a finite number `> 0`
(`is_static=True` needs no mass and ignores it). Any rejection - mass, size,
shape, an unloadable mesh - rolls the scene back and leaves the name reusable.

## Mesh objects

Beyond primitives, `add_object` can inject a triangle-mesh asset (STL/OBJ) into
the live scene at runtime. Pass `shape="mesh"` with a `mesh_path` to the asset
file; the extent is defined by the mesh's own units, so `size` is ignored on
this backend - a read the Isaac backend's mesh `add_object` shares. The Newton
backend consumes it instead, as a per-axis scale on the
loaded geometry, so a mesh add carrying a `size` does not mean the same thing
there - which meaning is right is tracked in
[#2300](https://github.com/strands-labs/robots/issues/2300).

```python
sim.add_object(name="bracket", shape="mesh", mesh_path="/abs/path/bracket.stl",
               position=[0.3, 0.0, 0.1])
# 'bracket' added: mesh at [0.3, 0.0, 0.1], extent=[0.12, 0.08, 0.03]m from the
# asset (collision uses its convex hull), 0.1kg
```

As for every shape, the success text reports the extent read back off the
compiled geom rather than echoing the request.

### A mesh geom collides as its convex hull

MuJoCo collides a mesh as its **convex hull**, not its triangles. A concave
asset (a room shell, a tray, a bowl) is filled solid for physics while the
camera still shows the open interior. For load-bearing concave geometry,
decompose the asset into convex parts and add one mesh object per part:

```python
for i, part in enumerate(convex_parts):          # e.g. a V-HACD decomposition
    sim.add_object(name=f"room_{i}", shape="mesh", mesh_path=part, is_static=True)
```

`mesh_path` is required for `shape="mesh"`; an unloadable file rolls the scene
back.

## Materials and textures

Pass `material=` to `add_object` to attach a MuJoCo material - matte or
textured - instead of the default glossy `color`; `color` still tints it.

```python
# Matte (non-plastic) surface: kill specular highlight + shininess.
sim.add_object("apple", shape="sphere", size=[0.04, 0, 0], color=[0.8, 0.1, 0.1, 1],
               material={"specular": 0, "shininess": 0, "reflectance": 0})

# Image texture from disk (absolute path), tiled 2x2 across the surface.
sim.add_object("table", shape="box", size=[0.5, 0.5, 0.02], is_static=True,
               material={"texture": "/abs/path/wood.png", "texrepeat": [2, 2],
                         "specular": 0, "shininess": 0})

# Procedural builtin texture (no image file needed).
sim.add_object("floor_tile", shape="box", size=[0.3, 0.3, 0.01], is_static=True,
               material={"builtin": "checker", "rgb1": [0.2, 0.3, 0.4],
                         "rgb2": [0.1, 0.2, 0.3], "texdim": 512})
```

`material` is a dict; all keys are optional:

| Key | Type | Meaning |
|-----|------|---------|
| `reflectance` / `specular` / `shininess` | float 0..1 | Surface response. `specular=0, shininess=0` = matte; the defaults read as glossy plastic. |
| `texrepeat` | `[u, v]` | Texture tiling across the surface. |
| `texture` | str | Absolute path to an image file (PNG/etc.) used as the RGB texture. |
| `builtin` | `"checker" \| "gradient" \| "flat"` | Procedural texture, coloured by `rgb1` / `rgb2` and sized `texdim` (default 512) per side. |

Specify **either** `texture` **or** `builtin`. An invalid path, an unknown
`builtin`, both at once, an unknown key (`rgb_1`), an empty dict or
`rgb1`/`rgb2`/`texdim` without `builtin` is refused (`ValueError`, a
`status=error` dict through the agent tool). MuJoCo only; Newton rejects a
non-`None` `material`.

## Surgical MJCF edits

`patch_scene_mjcf(ops)` applies a list of structured ops to the live spec and
recompiles once, preserving joint state, actuator setpoints and latched
wrenches. Each op accepts only the keys it reads:

| Op | Keys |
|----|------|
| `add_body` | `parent` (default `"world"`), `name` (required), `pos`, `quat` |
| `add_geom` | `body` (required), `type` (default `"box"`), `size`, `rgba`, `name`, `pos`, `quat` |
| `add_site` | `body` (default `"world"`), `name` (required), `pos`, `size`, `rgba` |
| `set_body_pos` | `name` (required), `pos` |
| `set_body_quat` | `name` (required), `quat` |
| `delete_body` | `name` (required) |

| field | accepted |
| --- | --- |
| `pos` | exactly 3 finite components |
| `quat` | exactly 4 finite components |
| `rgba` | 3 (RGB, completed with an opaque alpha) or 4 finite components |
| `size` | finite components, in the count the geom's shape consumes |

`add_geom`'s `type` takes the primitive shapes and refuses `"mesh"` (use
`add_object(shape="mesh", mesh_path=...)`). A three-component `rgba` is
completed with an opaque alpha. The batch is atomic: one refused op, or a
model MuJoCo will not build, rolls the world back to its pre-patch state.

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0.4, 0, 0.9]}])
# status=error: set_body_pos: unknown op key(s): 'position' (did you mean 'pos'?).
#               Accepted keys: name, op, pos.
```

Use `replace_scene_mjcf(xml)` for MJCF elements this vocabulary does not cover.

## Exporting a scene

`export_xml(output_path=...)` serialises the live scene, including every
runtime mutation, as MJCF that `load_scene` reloads:

```python
sim.export_xml(output_path="/tmp/handoff.xml")
other.load_scene(scene_path="/tmp/handoff.xml")   # same scene, same structure
```

Assets are referenced by absolute path, so the export reloads from any
location on the machine that produced it; copy the asset trees along with the
XML when moving it.

## Cameras

Free cameras look from `position` toward `target` (`fov=60.0`, `width=640`,
`height=480`); robot-URDF cameras are auto-discovered on `add_robot`. A
discovered camera is registered under its short name (`wrist`) and namespaced
(`so101/wrist`); a second robot's clashing short name is registered namespaced
instead. Enumerate with `list_cameras()` - see
[Simulation overview - Cameras](overview.md#cameras).

To mount a camera on a moving body pass `parent_body`, namespaced
`<robot>/<body>`; `list_bodies(robot_name=...)` returns every body plus
`gripper_body`, the best-guess end-effector mount:

```python
bodies = sim.list_bodies(robot_name="so101")["content"][1]["json"]
mount = bodies["gripper_body"]          # e.g. "so101/gripper" -- the wrist mount
sim.add_camera(name="wrist", parent_body=mount,
               position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1])  # local frame
```

A mounted camera survives `remove_robot` of other robots (it is re-mounted
after the rebuild); removing its own robot drops it with a warning.

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

- [Simulation overview](overview.md)
- [Domain randomization](domain-randomization.md)
