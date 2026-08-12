### Fixed: `set_geom_properties` refuses a size component a `<fromto>` fixes

A geom declared with `<fromto>` gives its extent along its own axis as two
endpoints, and the compiler fixes part of its `geom_size` row from them rather
than reading it from `size` - the axis extent, plus a box's or ellipsoid's
cross-section, which is made square from the first component. Resizing one of
those components reported success and then lost the change twice over: the value
was written into the spec's `size` row where the next scene recompile discarded
it, and the owning body's inertial row - re-derived from that same spec - kept
describing the extent the endpoints still declared. A `fromto` capsule resized
from a 0.15 m to a 0.30 m half-length therefore collided as the new shape while
resisting rotation as the old one, and reverted to 0.15 m at the next unrelated
`add_object`.

Such a change is now refused before either representation is touched, with a
message naming the component, the value the compiler keeps producing and how to
change it. The components a `fromto` leaves alone - a capsule's or cylinder's
radius - are unaffected and still recorded durably.
