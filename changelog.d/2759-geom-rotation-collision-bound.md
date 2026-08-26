### Fixed: a rotated collision geom is bounded on the axes it occupies

`_geom_aabb` reports the axis-aligned box one `<geom>` occupies in its owning body's frame, and
`load_mjcf_scene_objects` publishes the union of those as a `SceneObject`'s `size` and `position` -
the physics footprint that stands in for the object. It read `pos` and `size` and never asked how
the geom is turned. MJCF gives a geom five mutually exclusive ways to state a rotation (`quat`,
`euler`, `axisangle`, `xyaxes`, `zaxis`), and dropping it does not make the bound approximate, it
reports the extents on the wrong axes: a 0.60 m bar turned a quarter turn about z came back as
`(0.6, 0.04, 0.04)` where MuJoCo places it along y at `(0.04, 0.6, 0.04)`.

The same function already read a rotation on its other channel. A capsule or cylinder may spell its
placement with `fromto`, whose endpoints carry the axis, and `_segment_aabb` bounds that exactly. So
one geom got two different bounds depending on which of two spellings MuJoCo compiles to the same
shape was used for it.

Each primitive now gets its exact bound - the support function for a box and for an ellipsoid, and
for a capsule or cylinder the endpoints rebuilt from the rotation and handed to `_segment_aabb`, so
the two spellings share one answer rather than drifting. A sphere is unchanged, no rotation moving
it, and the `fromto` branch stays rotation-free because MuJoCo derives that geom's orientation from
the endpoints and discards any the element declares. `<compiler angle>` and `<compiler eulerseq>`
are forwarded from the scene entry point, which already resolved them for the mesh reader.

Over the 336 loadable MJCFs in the shipped asset cache, 421 bodies declare a rotated collidable
analytic geom and 27 of the 72 registered robots own at least one. Against MuJoCo's own geometry the
old rule under-bounded 279 of those bodies and over-bounded 247, the worst reporting 0.0085 m on an
axis the geometry spans 0.1601 m; afterwards none of the 421 is off by more than 0.1%.
