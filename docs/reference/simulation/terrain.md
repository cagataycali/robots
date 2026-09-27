---
description: Lay down a deterministic heightfield instead of a flat plane, and scale it with difficulty.
---

# Rough terrain

By default `create_world()` lays down a flat ground plane. A locomotion
policy is only interesting on ground it can trip on, so pass a `terrain=`
kind to lay down a deterministic heightfield instead - a floating-base robot
then settles onto and walks over it. Four kinds ship:

- `terrain="rough"` - smoothed value-noise bumps (robustness to uneven ground).
- `terrain="stairs"` - a flight of discrete step plateaus rising along +x
  (foot placement + climbing).
- `terrain="pyramid"` - concentric square step plateaus rising toward the centre
  from every direction (an omnidirectional climb).
- `terrain="slope"` - a constant-grade inclined ramp rising along +x
  (a continuous uphill pitch).

```python
sim.create_world(terrain="rough")        # bumpy heightfield ground
sim.add_robot("unitree_go2", keyframe="home")
```

A floating base is seated on the surface at `add_robot` and on every `reset()`,
and the seat is measured rather than assumed: the base is raised by the terrain
height beneath it and then by whatever its own geoms are still inside the ground.
That second term is what a height sample cannot see - the surface under a foot
0.3 m out is not the surface under the base, and a model's flat pose does not
always clear `z=0` (a LeKiwi's wheels sit 34.6 mm under its root body, a
straight-legged quadruped's feet 120 mm) - so an episode starts with the robot
resting on the terrain instead of being ejected out of it.

The field spans the same +/-5 m footprint as the flat plane (the reachable
workspace is unchanged), its surface ranges from 0 up to ~8 cm on a solid
base slab (flush with `z=0` at its lowest point, so a robot never falls
below the nominal floor), and it is regenerated identically on every
`reset()` (deterministic given the terrain kind), so a benchmark that
evaluates a policy on rough ground is reproducible. `terrain` only applies
when `ground_plane=True` (the default, which is the master floor switch);
an unknown kind is rejected with an error listing the supported kinds.
`ground_plane` itself must be a boolean: it selects a posture (lay a floor or
leave the world open), so a non-boolean is refused under the shared
`boolean_flag_error` domain rather than read by truthiness - `"false"` does
not lay a floor and `0` does not omit one (MuJoCo and Newton backends). It
is the ground-generation primitive a terrain *curriculum* (progressive
difficulty across resets) builds on. (MuJoCo backend; the Newton backend
rejects `terrain=` as not-yet-supported.)

Those guarantees - the field flush with `z=0` at its lowest cell, reaching the
full elevation at its highest, with the declared plateau count for a stepped
kind - are properties of the *grid* as much as of the kind, so each kind needs a
minimum number of cells to draw its shape at all. `create_world()` always uses a
40-cell grid and is comfortably above every minimum. A caller reaching for the
generator directly (`generate_heightfield(kind, resolution=...)`) is refused
below it, naming the kind and the count that works, rather than handed a field
that is flat or short of its top plateau; the minimums are exported as
`TERRAIN_MIN_RESOLUTION`.

That curriculum knob is `difficulty`, which scales the terrain's peak
elevation (the metre height its normalized `[0, 1]` field maps to) without
changing the terrain *kind*:

```python
sim.create_world(terrain="rough", difficulty=0.3)  # gentle bumps (early stage)
# ... later, harder stages ...
sim.create_world(terrain="rough", difficulty=1.0)  # full ~8 cm bumps (default)
sim.create_world(terrain="rough", difficulty=2.0)  # exaggerated ~16 cm bumps
```

`difficulty=1.0` (the default) is the full-height terrain, byte-identical to
omitting it; `<1` is gentler, `>1` harsher. It must be a finite *number*
`> 0` - the same positive-real domain every other continuous knob accepts, so
`0`, a negative value, `nan`/`inf`, a `bool` (`True` is not a scale, even
though it is an `int` subclass) and a string (including a numeric one like
`"0.5"`) are all refused with a structured error naming the parameter. Every
backend reports through that one domain, so a scale one `create_world` refuses
cannot be honored by another. It only applies with a `terrain` - setting
`difficulty != 1.0` on a flat world (no `terrain`) is rejected with an error
rather than silently having no effect. A locomotion curriculum ramps `difficulty` across resets to grow the
terrain the policy must handle.

A floating-base robot added to a terrain world (or reset in one) spawns
SEATED on the local terrain surface: its base is raised by the heightfield
height beneath its `(x, y)` so its feet rest on the ground, rather than at
the flat-ground keyframe height (which would leave them buried below a raised
heightfield). A flat ground plane and a fixed-base arm (no free joint) are
unaffected.

"Its base" is the robot's OWN floating base, resolved by ownership rather than
by name. That distinction matters when a robot's MJCF ships a free-jointed task
object of its own -- a payload, a kick ball, the grasping cube a Menagerie
manipulation scene declares under the robot's namespace. Such an object's joint
is a named entry in `robot_joint_names(...)` too, and on a mobile base whose own
`<freejoint>` is unnamed it is the only free joint that appears there at all, so
picking a base by name can land on the object. Seating never moves it: it is not
the robot's base, its `(x, y)` is not where the robot stands, and it is left
exactly where the scene put it. The same resolved base is what `get_observation`
reports as `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel` and what
`start_recording` declares those columns from, so the seated pose, the observed
pose and the recorded pose are the same body's.

## See also

- [World building](world-building.md)
- [Spawn pose and physics options](spawning.md)
