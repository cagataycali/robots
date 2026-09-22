---
description: Add primitives to a live scene - shape, size, mass - and the names they may not take.
---

# Objects

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

## Object shapes

`shape` takes one of seven values: `box`, `sphere`, `cylinder`, `capsule`,
`ellipsoid`, `plane` and `mesh`. All seven are offered by the agent-tool schema
too, so a model driving the simulation can select any of them. How many `size`
components each one consumes is in the table below.

## Object size

`size` is the **full extent in meters** along each local axis - not MuJoCo's
native half-extent. It is halved when the geom is compiled, so
`size=[0.05, 0.05, 0.05]` is a 5 cm cube.

Pass every component the shape consumes; a partial vector is rejected rather
than completed from a default, because a completed vector compiles a
differently-sized object while `add_object` reports success:

| Shape | Components consumed |
|-------|---------------------|
| `box` / `ellipsoid` | `[x, y, z]` - all three full edge lengths / diameters |
| `cylinder` | `[diameter, unused, full height]` - three (index 1 is ignored) |
| `capsule` | `[diameter, unused, cylinder-section length]` - three (index 1 is ignored). The two caps add `size[0] / 2` at each end, so the object stands `size[2] + size[0]` tall |
| `sphere` | `[diameter]` - one is enough |
| `plane` | `[x]` or `[x, y]` visual half-widths (`y` mirrors `x` when omitted) |
| `mesh` | none - the asset's own units define the extent |

At most 3 components are accepted; omit `size` entirely for the 5 cm default.

`add_object`'s success text reports the extent the geom **compiled to**, read
back off the model, never the request. The two agree only for `box` and
`ellipsoid`, the shapes that consume all three components; for every other row
in the table the request holds a value the geom does not carry, and echoing it
stated an extent the object does not have:

```python
sim.add_object("ball", shape="sphere", size=[0.05, 0.09, 0.2])
# 'ball' added: sphere at [0.0, 0.0, 0.0], size=[0.05, 0.05, 0.05], 0.1kg
#   the ball is 5 cm across in every axis; 0.09 and 0.2 described nothing
sim.add_object("rod", shape="capsule", size=[0.05, 0.0, 0.9])
# 'rod' added: capsule at [0.0, 0.0, 0.0], size=[0.05, 0.05, 0.95], 0.1kg
#   0.95 m tall, not the 0.9 m asked for - the caps add the diameter
sim.add_object("floor", shape="plane", size=[1.0, 2.0], is_static=True)
# 'floor' added: plane at [0.0, 0.0, 0.0], size=[1.0, 2.0] visual half-widths
#   (infinite for collision), static
```

`set_geom_properties(size=...)` resizes an existing geom and takes a *different*
convention for the same word: the compiled geom's own MuJoCo `geom_size`
components. The two are not interchangeable - `size=[0.2, 0.2, 0.2]` builds a
20 cm box here and resizes that same box to 40 cm there, and this table's
`[diameter, unused, height]` capsule triple is refused there (it wants
`[radius, half-length]`). See
[Domain randomization](domain-randomization.md).

```python
sim.add_object("crate", shape="box", size=[0.5])
# status=error: box needs 3 'size' component(s) [x, y, z] full edge lengths,
#               got 1 (size=[0.5]). ...
sim.add_object("crate", shape="box", size=[0.5, 0.5, 0.5])   # 50 cm crate
```

Every component must also be a finite number, and **that** part of the domain is
shared with the Newton and Isaac backends' `add_object` - word for word, not just
verdict for verdict - so an extent one backend refuses is refused by all three
with the same message. A `nan`/`inf`, boolean, `None` or otherwise non-numeric
component is rejected by name rather than reaching the solver, a NumPy array is
accepted and normalized to plain floats, and a value that is not a vector at all
is refused instead of raising from whatever first tries to iterate it:

```python
sim.add_object("crate", shape="box", size=[float("nan"), 0.1, 0.1])
# status=error: add_object: 'size' must contain finite numbers (no nan/inf),
#               got [nan, 0.1, 0.1]
sim.add_object("crate", shape="box", size=0.5)
# status=error: add_object: 'size' must be a list/tuple of numbers, got 0.5
sim.add_object("crate", shape="box", size=np.array([0.5, 0.5, 0.5]))   # accepted
```

An **empty** `size` is a component count, not an omission, so it is rejected
rather than quietly taking the default extent - omit `size` (or pass `None`) to
ask for the default. Here the three backends agree on the verdict but not on the
wording, because MuJoCo reaches an empty vector through the per-shape count above
and so names the count the shape needs:

```python
sim.add_object("crate", shape="box", size=[])
# status=error: box needs 3 'size' component(s) [x, y, z] full edge lengths,
#               got 0 (size=[]). ...
```

The per-shape counts in the table above remain MuJoCo's alone. Newton and Isaac
accept a short `size` (Isaac documents completing the missing trailing components
from defaults), and neither bounds a component to be positive, so a vector this
backend refuses on either of those axes may still be accepted there. Converging
the three is tracked in
[#1858](https://github.com/strands-labs/robots/issues/1858).

## Object mass

`mass` (kg) applies to dynamic objects and must be a finite number greater than
zero - the same domain `set_body_properties(mass=...)` enforces when it writes
the same body, and the same one the Newton and Isaac backends' `add_object`
applies, so a mass one backend refuses is refused by all three. A mass outside it
is rejected up front, naming the parameter, instead of surfacing as a recompile
failure:

```python
sim.add_object("crate", shape="box", mass=0)
# status=error: add_object: 'mass' must be a finite number > 0, got 0.0
sim.add_object("crate", shape="box", mass=1e-16)
# status=error: add_object: 'mass' must be >= MuJoCo's mjMINVAL (1e-15 kg) ...
```

This matters beyond the one object: a body's mass divides every force acting on
it, and the solver keeps a single state vector, so an infinite mass turns the
whole world's `qpos`/`qvel` to `nan` on the next step - every other body
included. `is_static=True` needs no mass (MuJoCo derives it from the geom's
density), so `mass` is ignored there - and not validated, on any backend, since
nothing reads it. The Newton backend additionally documents `mass=0` as an
alternative spelling of `is_static=True` and keeps accepting it; MuJoCo and Isaac
refuse a zero mass and name that flag as the remedy.

Whatever the reason for a rejection - mass, `size`, an unsupported `shape`, an
unloadable mesh - the scene is rolled back to its previous compilable state and
the object name stays reusable, so a corrected retry under the same name works
and one bad add never bricks later scene edits.

## Object names do not collide with robot labels

A robot's label is not one of its body names - those are `<label>/base`,
`<label>/gripper`, ... - so MuJoCo's repeated-name check never saw an object
named after a robot in the world, while every by-name reader did: the object
took over `get_body_state(body_name=...)`, `add_camera(parent_body=...)` and
`attach_bodies(parent=...)` for the arm the caller meant, reporting success
each time. Both directions are refused before anything is registered:

```python
sim.add_robot(name="so101", data_config="so101")
sim.add_object("so101", shape="box", size=[0.05, 0.05, 0.05])
# status=error: add_object: 'so101' is the name of a robot in this world, and an
#               object under that name would answer get_body_state /
#               attach_bodies / add_camera calls meant for the robot (its bodies
#               are 'so101/<body>'; see list_bodies). Pick another name.

sim.add_object("cube", shape="box", size=[0.05, 0.05, 0.05])
sim.add_robot(name="cube", data_config="so101")
# status=error: Robot name 'cube' is already an object in this world; by-name
#               reads (get_body_state, attach_bodies, add_camera) would keep
#               resolving to that object. Pick another name, or omit name= to
#               auto-number.
```

That last remedy is one you can take: the label `add_robot` derives when
`name` is omitted skips names held by objects as well as by robots, so the
short form stays usable in a world where an object already carries the model's
name.

```python
sim.add_object("so101", shape="box", size=[0.05, 0.05, 0.05])
sim.add_robot(data_config="so101")   # no name= : derives 'so101_2', not a clash
```

An object named after an existing *body* was already refused by MuJoCo
("repeated name") and still is.

## See also

- [World building](world-building.md)
- [Meshes and materials](meshes-and-materials.md)
