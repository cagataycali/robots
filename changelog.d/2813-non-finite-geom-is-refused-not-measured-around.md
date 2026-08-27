### Fixed: a geom declaring a placement or extent that is not finite is refused, not measured around

`_geom_aabb` folds each of a body's geoms into the axis-aligned box
`load_mjcf_scene_objects` reports as a `SceneObject`'s `size` and `position` - the collision proxy
the Isaac realization stands the object up with. `_body_collision_aabb` unions those with a running
`min`/`max`, and Python orders a NaN as neither smaller nor larger than anything, so the comparison
keeps the accumulator it started with. The geom carrying it disappeared from the bound and nothing
reported that.

A static fixture whose leg declares `euler="nan 0 0"` was measured as its tabletop alone: `size
(0.8, 0.6, 0.04)` where the file declares `(0.8, 0.6, 0.77)` - 19.2x too short, and byte-identical
to the same fixture with the leg deleted, under a successful load. Each spelling failed differently:
`pos="nan ..."` dropped the geom on the axis that was not finite and kept it on the others;
`pos="inf ..."` reported an infinite centre and a NaN extent; `size="inf ..."` reported a NaN centre
and, because `inf - inf` is a NaN that `_recursive_collision_aabb`'s own accumulator drops in turn,
sized that axis at the `1e-4` floor - the smallest proxy the reader can emit, for the largest geom
the file can declare.

Reachability is specific. MuJoCo refuses a non-finite geom on a body with a free joint, because the
inertia it derives is then not finite; it *compiles* the same geom on a body without one - and a
body without a free joint is exactly what this loader calls a fixture, the tables and cabinets whose
footprint a manipulation scene is planned against. So the affected scenes are precisely the ones
that load, which is why the reader cannot defer the question to a compile step.

MuJoCo is also the oracle for the disposition: it refuses a non-finite geom quantity wherever it
checks one (`nan size in geom`) and warns about the document as a whole (`XML contains a 'NaN'`), so
refusing is the format's own answer rather than a policy invented here. It is also what this
module's stated failure semantics already promise - loaders never silently return a phantom robot.
The same defect class was closed one module over for mesh assets, whose refusal describes this
mechanism in the same words; `loaders.py` carried no finiteness guard at all.

One guard covers all four parse paths - `pos`, `size`, whichever orientation spelling the geom used,
and `fromto` - so no one spelling drifts into tolerating what its siblings refuse, and the refusal
names both the offending attribute and the geom. Only a value that *parsed* is graded, so an
unreadable attribute keeps falling back to its documented default exactly as before, and a geom with
no analytic AABB still returns `None` so the caller falls back to another geom. Over the shipped
corpus - 542 MJCF documents, 14940 `<geom>` elements - zero declare a non-finite placement or
extent, so no shipped asset changes behaviour.
