### Fixed: a body's collision proxy is bounded by the geoms MuJoCo can collide

`_body_collision_aabb` computes the axis-aligned box that stands in for a body's physics
footprint, and it is what `load_mjcf_scene_objects` reports as a `SceneObject`'s `size` and
`position`. It selected geoms with `group == "0"`, falling back to every geom when that matched
nothing, and that is wrong in two independent ways.

`group` carries no contact meaning in MuJoCo - it is a visualiser toggle - and the conventions
built on it disagree, MuJoCo Menagerie marking collision geoms with `group="3"` where robosuite
marks visual ones with `group="1"`. So the attribute cannot answer which geoms the solver will
touch, which is the only question this function asks. Separately, the comparison was against the
literal string: a geom that omits `group` *is* group 0, and MuJoCo resolves it to `geom_group == 0`,
but the attribute reads as absent so the `"0"` pass skipped it.

Both directions are measurable. In a Menagerie-shaped body every geom falls through to the
every-geom fallback, so the proxy becomes the union of the decorative shell and the collision
primitive: the shipped `franka_emika_panda` `mjx_hand` fingers reported a
`(0.022, 0.026, 0.0532)` proxy centred `(0, 0.013, 0.0269)` where the body's collision geometry is
`(0.0175, 0.0152, 0.0165)` centred `(0, 0.0076, 0.04525)` - 6.9x the volume, 3.2x the length along
the finger, and centred on the wrong part of it. Of that finger's seven geoms exactly one,
`left_finger_pad`, is left able to take part in a contact (`conaffinity="3"`); not one of them
carries `group="0"`. And where the `"0"` pass did match, it narrowed instead: a body whose two
collidable geoms spell `group="0"` and `group="3"` dropped the second one entirely, reporting a
bound that excludes geometry the solver collides.

The signal read is now the format's own, through the existing `_geom_cannot_collide` - already the
tie-break `_mesh_geom_visual_rank` ranks first for the mirror question of which mesh is a body's
visual asset. MuJoCo lets two geoms touch only when `contype1 & conaffinity2` or
`contype2 & conaffinity1` is non-zero, so a geom declaring both as `0` can never take part in a
contact. Reading either half of that pair alone would be wrong: `contype="0" conaffinity="1"` still
collides with anything declaring `contype` non-zero, and such a geom stays inside the bound.

Graded against MuJoCo itself across the downloaded asset corpus - 336 compilable MJCF files, 427
bodies whose collidable analytic geoms are axis-aligned and therefore comparable to an axis-aligned
bound - the reported bound now matches MuJoCo's own on 427 of 427, against 356 of 427 before. No
body is worse. Over the wider corpus of 554 files and 1209 bodies with an analytic answer, 72
change, all 72 shrink, none grows, and 27 also move their centre onto the geometry they stand for.

Deliberately unchanged: a body whose every analytic geom is contact-free. There is nothing
collidable to prefer, so all of them are bounded exactly as before - an approximate proxy is still
better than none - and that is pinned as a control, as is the single-geom case.

The `group` tier is dropped here rather than demoted the way `_mesh_geom_visual_rank` keeps it.
That is a measurement, not a preference: keeping it as a refinement inside the collidable tier
changes the answer for 0 of those 1209 bodies, whereas for the visual-mesh question it is the only
signal some files carry.
