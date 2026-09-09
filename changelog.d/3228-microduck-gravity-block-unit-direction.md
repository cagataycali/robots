### Fixed: the Microduck gravity block is the unit direction its layout declares

Slot two of the Microduck observation vector is documented as "a UNIT gravity
direction in the base frame", and `build_observation` held `base_quat` to a width
and to finiteness on the way there but not to describing a rotation.
`quat_rotate_inverse` evaluates a formula that is a rotation only for a unit
quaternion - it mixes a term quadratic in the components with a linear one, so
scaling does not cancel. A quaternion scaled by any positive factor encodes the
same rotation, which is why the library's single orientation domain
(`coerce_orientation_quaternion`) accepts any magnitude, on the stated ground
that "every consumer either normalizes or is scale-invariant"; this was the one
consumer that did neither. A 20 deg pitch offered at `|q| = 2` reached the graph
as a 61 deg attitude of magnitude 1.56, and an all-zero quaternion - what an
orientation that was never written or was dropped on the wire spells - was
answered with world `-Z` unchanged, which is the gravity block of a perfectly
upright base.

The orientation is now normalized inside `quat_rotate_inverse`, and a norm below
the shared `MIN_QUATERNION_NORM` floor is refused rather than read, matching the
two same-layer siblings (`policies/wbc/control.quat_rotate_inverse`,
`policies/protomotions/state_utils.quat_rotate_inverse`). A `nan`/`inf`
component still flows through so the assembled-vector pass names it against the
block it becomes, and for an exactly-unit quaternion the pre-existing expression
is reproduced bit for bit.
