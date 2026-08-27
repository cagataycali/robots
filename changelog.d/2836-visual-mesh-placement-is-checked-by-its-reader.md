### Fixed: an Isaac scene object no longer reports a visual mesh placement that is not a placement

`load_mjcf_scene_objects` reports two placements per object. The collision proxy -
`position` and `size` - comes from `_recursive_collision_aabb`. Where the visual
asset hangs on that proxy comes from `_find_body_mesh`, which returns the
`mesh_pos` and `mesh_quat` a `SceneObject` carries, and read both straight through
`_parse_xyz` / `_parse_orientation`.

`_geom_aabb` refuses a non-finite geom `pos`, `size`, orientation or `fromto` at
the parse, and never saw the geom this reader picks. `_body_collision_aabb` walks
`for collidable_only in (True, False)` and returns as soon as a pass finds a
bound, so on a body that owns a collision primitive the second pass never runs and
a contact-free geom is never handed over; a `type="mesh"` geom has no analytic AABB
in any case. `_find_body_mesh` prefers precisely that geom, because MuJoCo
Menagerie marks a visual geom `contype="0" conaffinity="0"` and declares the
collision geom first. The one geom whose placement was reported was the one geom
whose placement was unchecked.

One of four body shapes leaked, and it is the Menagerie one - a contact-free mesh
geom beside a collidable sibling reported `mesh_pos=(0.0, 0.0, nan)` under a
successful load, while the other three were already refused. `position` and `size`
stayed finite, because the accumulator supplies them from the collision primitive,
so the object's physics read healthy while its visual asset was hung at a
coordinate that is not a coordinate - the wrong half to lose for a reader that
exists to prefer the asset a pixel-conditioned policy was trained on. A single bad
component was not contained either: `_parse_orientation` normalizes, so
`quat="1 0 nan 0"` reported `mesh_quat=(nan, nan, nan, nan)`.

Both quantities now go through `_refuse_non_finite_geom`, the module's existing
owner for this wording, at the parse rather than at the return - the contact-free
set is exactly the strongest-ranked set, so the geom the bound cannot vouch for is
always the one the reader picks. An attribute that cannot be parsed still falls
back to its documented default, finite values are untouched however extreme, and
the nested-body `pos` this reader folds into its offset is left to
`_recursive_collision_aabb`, which refuses it on the same `findall("body")`
traversal. Of the 570 MJCF files cached on the measuring machine, 2053 of 3054
bodies are in the leaking shape - every link of the `so101` the registry ships
included - and none carries a non-finite value today, so the defect was latent;
60 of 60 resolvable registry assets load unchanged.
