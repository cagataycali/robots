---
description: Patch the live MJCF spec with structured ops, and serialise the scene back out.
---

# Editing and exporting a scene

## Surgical MJCF edits

`patch_scene_mjcf(ops)` applies a list of structured ops to the live spec and
recompiles once, preserving joint state for untouched joints. Each op accepts
only the keys it reads:

| Op | Keys |
|----|------|
| `add_body` | `parent` (default `"world"`), `name` (required), `pos`, `quat` |
| `add_geom` | `body` (required), `type` (default `"box"`), `size`, `rgba`, `name`, `pos`, `quat` |
| `add_site` | `body` (default `"world"`), `name` (required), `pos`, `size`, `rgba` |
| `set_body_pos` | `name` (required), `pos` |
| `set_body_quat` | `name` (required), `quat` |
| `delete_body` | `name` (required) |

Any other key is rejected. Every field above has a fallback default (`pos` the
origin, `quat` identity, `type` `"box"`, `parent` the worldbody), so a key the op
does not read is not inert - it would leave that default in place while the patch
reports success:

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0.4, 0, 0.9]}])
# status=error: set_body_pos: unknown op key(s): 'position' (did you mean 'pos'?).
#               Accepted keys: name, op, pos.
```

Every numeric field an op writes is held to the domain the scene-construction
calls apply to the same buffer:

| field | accepted |
| --- | --- |
| `pos` | exactly 3 finite components |
| `quat` | exactly 4 finite components |
| `rgba` | 3 (RGB, completed with an opaque alpha) or 4 finite components |
| `size` | finite components, in the count the geom's shape consumes ([Object size](objects.md#object-size)) |

`add_geom`'s `type` takes the primitive shapes - `box`, `capsule`, `cylinder`,
`ellipsoid`, `plane`, `sphere` - and refuses `"mesh"`: the op has no key that
could name a mesh asset, so the geom would have no mesh to take its extent from
and MuJoCo would refuse the whole scene at recompile. Add a mesh through
`add_object(shape="mesh", mesh_path=...)`, which registers the asset alongside
the body:

```python
sim.patch_scene_mjcf([{"op": "add_geom", "body": "rig", "type": "mesh"}])
# status=error: add_geom: 'type' cannot be 'mesh' - this op has no key that names
#               a mesh asset ... Add a mesh with add_object(shape="mesh",
#               mesh_path=...), which registers the asset alongside the body.
```

MuJoCo bakes a `nan`/`inf` component into the model without complaint, so an
unchecked one reports success and only surfaces later as a poisoned physics
state. A wrong component count is reported by the library rather than left to
MuJoCo, which for the two attribute-assigning ops (`set_body_pos`,
`set_body_quat`) dumps a C++ overload table naming neither the op nor the field:

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [float("nan"), 0, 0.3]}])
# status=error: set_body_pos: 'pos' must contain finite numbers (no nan/inf),
#               got [nan, 0, 0.3]

sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [0.4, 0.9]}])
# status=error: set_body_pos: 'pos' must be a 3-element vector, got 2 ([0.4, 0.9])
```

A three-component `rgba` is the same RGB `add_object(color=...)` accepts, so the
two surfaces that write `geom_rgba` agree on what a colour is:

```python
sim.patch_scene_mjcf([{"op": "add_geom", "body": "rig", "type": "box",
                       "size": [0.1, 0.1, 0.1], "rgba": [0.9, 0.3, 0.1]}])
# status=success - stored as [0.9, 0.3, 0.1, 1.0]
```

The batch is atomic: if any op is rejected the world is rolled back to its
pre-patch state, so a bad key or a non-finite component never leaves a
half-applied scene. A batch every op accepts can still be refused by MuJoCo when
the model they add up to is one it will not build, and that refusal is rolled
back on the same terms - it costs the batch, not the world, so the next mutation
still succeeds.

A successful batch recompiles the model once, so it keeps the dynamic state every
other scene mutation keeps: joint positions and velocities, actuator setpoints,
and a latched `apply_force` wrench. Use
`replace_scene_mjcf(xml)` for MJCF elements this vocabulary does not cover.

## Exporting a scene

`export_xml(output_path=...)` serialises the live scene - including every runtime
mutation - as MJCF. It is the read sibling of `replace_scene_mjcf`, so the file it
writes is meant to be reloadable:

```python
sim.export_xml(output_path="/tmp/handoff.xml")
other.load_scene(scene_path="/tmp/handoff.xml")   # same scene, same structure
```

An absolute `output_path` is written as given. A relative one - `"scene.xml"`,
`"handoff/scene.xml"` - lands under `~/.strands_robots/scenes/`
(`STRANDS_ROBOTS_SCENE_ROOT`), not the process working directory, so an agent
asked to "save the scene" does not drop files into whatever directory the process
was started from; the success text names the resolved path either way.

`load_scene` reads that same directory, so the round trip holds under one bare
name - the spelling an agent actually uses:

```python
sim.export_xml(output_path="handoff.xml")          # -> ~/.strands_robots/scenes/handoff.xml
other.load_scene(scene_path="handoff.xml")         # same file, found there
```

A `scene_path` is read AS GIVEN first, so a relative path that resolves against
the working directory today keeps resolving there; the scenes directory is
searched only when nothing is at the path the caller spelled. A file that is in
neither is refused with both directories named.

Mesh, texture and height-field assets are referenced by ABSOLUTE path. MuJoCo
resolves a relative `file=` against the model's own directory (plus `meshdir` /
`texturedir`, or the `assetdir` that sets both), and that directory is not part
of the serialised XML - so a
relative reference would resolve against wherever the export happened to be
written. Absolute references keep the export reloadable from any location, and a
scene composed from several models needs them: each model contributes assets from
its own root, so no single `meshdir` could cover them all.

The consequence is that an export names paths on the machine that produced it.
Copying the XML alone to another machine will not carry the assets with it;
copy the referenced asset trees too, or re-compose the scene there from the same
`add_robot` calls.

## See also

- [World building](world-building.md)
- [Objects](objects.md)
