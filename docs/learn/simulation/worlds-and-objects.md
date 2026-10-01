---
description: Build a scene on any backend: objects and their size conventions, cameras, articulated task objects, MJCF patches, meshes, materials, terrain.
---

# Worlds and objects

By the end of this page you can build a scene from primitives, an articulated carton the predicates read, a wrist camera riding the arm, an MJCF patch and stairs under it all, with the same calls on every backend.

```python
from strands_robots.simulation import create_simulation
from strands_robots.simulation.task_objects import list_task_objects, task_object_path

sim = create_simulation("mujoco")
sim.create_world(terrain="stairs", difficulty=0.5)
sim.add_robot("so101", position=[0.0, 0.0, 0.0])
sim.add_object(name="cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.25, 0.0, 0.2], color=[1, 0, 0, 1])
sim.add_object(name="ball", shape="sphere", size=[0.04], position=[0.25, 0.1, 0.3], mass=0.05)
sim.add_object(name="shelf", shape="box", size=[0.2, 0.1, 0.02], position=[0.4, 0.0, 0.3], is_static=True,
               material={"builtin": "checker", "rgb1": [0.8, 0.8, 0.8], "rgb2": [0.2, 0.2, 0.2], "texrepeat": [4, 4]})
print(list_task_objects())
sim.add_robot(name="carton", urdf_path=task_object_path("hinged_carton"), position=[0.35, -0.2, 0.0])
print(sim.robot_joint_names("carton"))
sim.add_camera(name="front", position=[0.8, 0.0, 0.6], target=[0.3, 0.0, 0.1])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
sim.patch_scene_mjcf([
    {"op": "add_body", "parent": "world", "name": "post", "pos": [0.5, 0.3, 0.1]},
    {"op": "add_geom", "body": "post", "type": "cylinder", "size": [0.04, 0.04, 0.2], "rgba": [0.2, 0.2, 0.8]},
])
sim.step(300)
sim.move_object("cube", position=[0.3, 0.0, 0.2])
print(sim.list_objects()["content"][0]["text"])
print(sim.get_ground_height(1.0, 0.0)["content"][0]["text"])
sim.cleanup()
```

You should see:

```text
['hinged_carton', 'open_tray', 'sliding_carton']
['cap_hinge']
Objects:

  - cube: box at [0.3, 0.0, 0.2], 0.1kg
  - ball: sphere at [0.25, 0.1, 0.0396], 0.05kg
  - shelf: box at [0.4, 0.0, 0.3], static
Ground height at (1.0000, 0.0000) = 0.0240m
```

## The world

`create_world(timestep=None, gravity=None, ground_plane=True, terrain=None, difficulty=1.0)` builds a `SimWorld`: `robots`, `objects`, `cameras`, `timestep` (0.002 s), `gravity`, `terrain`, `status`, `sim_time`, `step_count`. `reset()` restores every robot's spawn pose (`keyframe` included) and every object's position; `destroy()` drops the world.

## Objects

`add_object(name, shape="box", position, orientation, size, color, mass=0.1, is_static=None, mesh_path=None, material=None)`. Shapes: `box`, `sphere`, `cylinder`, `capsule`, `ellipsoid`, `plane`, `mesh`. On MuJoCo `size` is the full extent in metres, halved internally:

| shape | `size` |
|---|---|
| `box`, `ellipsoid` | `[x, y, z]` full edge lengths |
| `sphere` | `[diameter]` |
| `cylinder` | `[diameter, unused, full height]` (three components) |
| `capsule` | `[diameter, unused, cylinder length]`; total height is `size[2] + size[0]` |
| `plane` | visual half-widths; infinite for collision, forced static |
| `mesh` | ignored; the file's units define the extent, `mesh_path` required |

A short vector is refused, not padded: padding would compile a different object and pass. `color` is RGB or RGBA; an RGB triple gets an opaque alpha. `is_static` is tri-state: `None` lets the backend decide (a plane is always static), `True` welds, `False` is a free body; a non-boolean is refused. `material` accepts `builtin` (`checker`, `gradient`, `flat`), `rgb1`, `rgb2`, `texrepeat`, `texdim`, `texture`, `reflectance`, `shininess`, `specular`; anything else is refused with the accepted list. Newton consumes half-extents and radii directly; Isaac pads trailing components from a documented default.

`move_object(name, position, orientation)` places a dynamic object at rest or rebuilds a static one. `remove_object`, `list_objects` and `get_body_state` complete the set. `attach_bodies(parent, child, mode="weld")` and `detach_bodies(parent, child)` glue two bodies at their current pose.

## Cameras

`add_camera(name, position, target, fov=60, width, height, parent_body=None)`; `fov` is vertical on every backend. World-frame by default. With `parent_body` the camera rides a body and `position` and `target` are in that body's frame, both required, since the world-frame defaults would put a wrist camera 1.7 m away. Every backend supports it. A name is a bare token (`wrist`, `front_cam`, `cam-2`), optionally scoped to one robot (`arm0/wrist_cam`); a space (`a b`), a dot (`wrist.rgb`) or `..` is refused with `status="error"` and the scene continues without the camera, so read the status. Every camera appears in `get_observation` as an `(H, W, 3)` array under its name; the policy pages explain [which names a model expects](../policies/lerobot-local.md).

## Task objects

Three MJCF assets ship in `strands_robots/simulation/task_objects/`: `hinged_carton` (lid on a hinge, joint `cap_hinge`, radians), `sliding_carton` (lid slides, joint `cap_slide`, metres), `open_tray` (rigid receptacle, no joints). Load one with `add_robot(name=..., urdf_path=task_object_path(...))`; its joints are namespaced under `name/` and visible to `joint_above`, `joint_below` and `joint_progress`. Contents are not in the assets; spawn small spheres per task and score them with `particles_inside` and `particles_spilled`.

## Scene editing

`patch_scene_mjcf(ops)` edits the live `MjSpec` atomically with `add_body`, `add_geom`, `add_site`, `set_body_pos`, `set_body_quat`, `delete_body`. Each op accepts only the keys it reads; a misspelled key is refused with a close match, since every field has a default and a typo would otherwise pass. `pos` is three finite numbers, `quat` four, `rgba` three or four. `replace_scene_mjcf(xml)` swaps the whole model, `load_scene(path)` starts from a file, `export_xml(path)` writes the current one. `SpecBuilder` in `mujoco/spec_builder.py` is the lower-level builder these use.

## Meshes and materials

`add_object(shape="mesh", mesh_path="part.stl", mass=0.2)` registers the file as a MuJoCo mesh asset (`spec.add_mesh`), so any format MuJoCo's compiler reads works; a mesh `size` scales per axis on Newton only; MuJoCo and Isaac ignore it and report success (#2300); the file's own units set the extent. Robot meshes come from `robot_descriptions` and the bundled menagerie tree through `strands_robots.simulation.model_registry` (`resolve_model`, `register_urdf`, `list_available_models`). Isaac converts MJCF and meshes into USD (`isaac/mjcf_assets.py`, `isaac/mesh_assets.py`).

## Terrain

`create_world(terrain="rough" | "stairs" | "pyramid", difficulty=1.0)` replaces the flat plane with a height field: `rough` is smoothed value noise, `stairs` five discrete steps rising along +x, `pyramid` concentric square plateaus. The field is 10 m across at 25 cm cells and seeded (`TERRAIN_SEED = 0`), so `reset()` regenerates it identically. `get_ground_height(x, y)` reads it, and the locomotion predicates (`base_height`, `base_below_z`) measure against it. Constants: `strands_robots/simulation/terrain.py`.
