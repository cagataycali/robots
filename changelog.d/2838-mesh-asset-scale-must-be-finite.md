### Fixed: an Isaac scene object no longer reports a mesh bound measured through a scale that is not a scale

`load_mesh_geometry` already refuses a vertex coordinate that is not finite, and
`_non_finite_vertex_error` states why: `min`/`max` order a NaN as neither smaller
nor larger than anything, so the running comparison `mesh_aabb` uses "drops it and
returns the bounds of the vertices that are finite - the same numbers a mesh
declaring only those would produce, under no error at all". That rule was applied
where the four parse paths meet, so both consumers of the file's own coordinates
agree about the same asset.

`scale` reaches the same comparison and did not go through it. `mesh_aabb` measures
`vertex * scale`, so a non-finite component poisoned every vertex on that axis and
the comparison dropped all of them, leaving the axis at its `inf`/`-inf` seed. The
asset's coordinates being finite was not enough: the transform applied to them
arrives from the caller rather than from the file.

The reachable route is an MJCF `<asset><mesh scale=...>`, parsed by
`_parse_mjcf_mesh_assets` through `_parse_axis` - which returns three floats, so
`nan` and `inf` pass. MuJoCo compiles such a model, so the reader could not defer
the question to the format's owner, and `load_mjcf_scene_objects` measures the
mesh's own bounds for a *mesh-only* body: the `elif mesh_path is not None` branch
its own comment says exists "so a mesh-only body can fall back to the mesh's own
bounds". A 0.2 m asset was then reported at `position=(nan, nan, nan)` with
`size=(0.0001, 0.0001, 0.0001)` under a successful load.

Neither failure is screenable from the reported fields, and the NaN one is the
worse half. Because the reported extent is floored at `1e-4`, a NaN axis leaves the
size **finite** - a plausible 0.1 mm, 2000x smaller per axis than the asset - so a
consumer screening the reported fields for non-finite values catches the centre and
not the size. An infinite axis reports an infinite centre and a size of NaN,
because the extent is `inf - inf`. A body that also carries collidable geometry
never reaches the measurement, and its reported bound was already correct, which is
what made the leak specific to the branch that exists to prefer the asset itself.

`mesh_aabb` now refuses a non-finite `scale` through `_non_finite_scale_error`, a
sibling to the vertex wording rather than a reuse of it: a bad coordinate in the
file and a bad transform from the caller are two causes, so they are two messages.
The check precedes the parse, because a transform that can never be applied is not
worth reading a mesh for and the check needs nothing from the file; a caller whose
path is wrong and whose scale is fine still gets the file contract's own message.

A finite scale is untouched, including the two spellings that look degenerate and
are not: `0.0` is a finite request to flatten an axis, where the `1e-4` floor is
doing the job it is there for, and a negative scale mirrors the asset, where the
running comparison is order-independent and the bound is already right. Both are
real: of the 17322 `<mesh>` entries across the 811 MJCF models cached on the
measuring machine, 1631 declare a non-unit scale and those include `(-1, -1, -1)`,
`(-0.001, 0.001, 0.001)` and millimetre scales - and **none** is non-finite, so the
defect is latent. 9 of the 59 loadable registry robots reach this measurement, one
of them through a millimetre scale, and all 60 loadable scenes report byte-identical
objects before and after.
