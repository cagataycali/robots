### Fixed: an Isaac scene object's `fromto` collision geom carries its length and its offset

MJCF spells a capsule or cylinder two ways: `pos` plus
`size="radius half-length"`, or `fromto` plus `size="radius"`, where the two
endpoints carry both the placement and the axis extent. The Isaac scene-object
loader read only the first spelling, so a `fromto` geom's single `size`
component fell through to the one-size `(r, r, r)` fallback and the absent
`pos` put the box on the body origin. A 0.60 m `fromto` capsule was reported as
a 0.05 m cube at the origin - 7.7% of the object's long axis, 125 cm3 against
1625 cm3, and 0.30 m from the object's centre - as `status="success"`. The
offset is the same value the LIBERO pose applier adds when it places the prim,
so the placement inherited the error too. `_geom_aabb` now consults `fromto`
first for a capsule or cylinder, taking the segment midpoint as the centre and
`|d|/2 + r` as the half-extent along each axis, with the `+ r` applied to the
cross-section only for a cylinder, whose flat caps do not extend past the end
discs. This matches what MuJoCo's own `geom_aabb` reports for the same geom,
and it mirrors a rule the MuJoCo backend already owns in
`scene_ops.fromto_fixed_size_components`.

The one-`size`-no-`fromto` fallback is dropped rather than kept: that spelling
is not a shape MuJoCo compiles (`size 1 must be positive in geom`), so the old
`(r, r, r)` box was the one answer no compilable input could produce. `fromto`
on a box or an ellipsoid additionally squares the cross-section and keeps the
existing `pos` + `size` reading.
