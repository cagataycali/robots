### Fixed: a non-finite MJCF mesh scale is refused on every body that carries it

`mesh_aabb` refuses a `<mesh scale=...>` whose components are not all finite,
because it measures `vertex * scale` and `min`/`max` order a NaN as neither
smaller nor larger than anything. `load_mjcf_scene_objects` reaches that
measurement only on the branch where no collidable analytic geom supplied the
bound - yet it carries the scale onto the `SceneObject` on every branch, and the
Isaac realization applies it to the visual prim's xform.

So a body declaring both a collidable geom and a visual mesh took its bound from
the geom, never measured, and reported `mesh_scale=(nan, 1.0, 1.0)` under
`success`. `position`, `size` and `offset` all stayed finite and correct, because
the collidable geom supplied them, so every field a consumer would screen for a
non-finite value was healthy. 53 of the 63 mesh-carrying bodies in the shipped
registry are in that shape.

The value lands in one `_author_local_xform(translate=..., orient_wxyz=...,
scale=...)` call whose other two arguments come from `mesh_pos` and `mesh_quat`,
both already refused when they are not finite. The finiteness test now lives in
`refuse_non_finite_scale`, the single owner both consumers call, and the loader
reaches it where the scale is read - beside the missing-file contract, for the
same reason: an asset declaration a body reaches has to be usable. The refusal's
wording gains the xform consequence alongside the sizing one.

A finite scale is untouched, including the 250 non-unit and the negative and zero
ones the shipped corpus declares. All 60 resolvable registry models still load,
and none declares a non-finite scale anywhere.
