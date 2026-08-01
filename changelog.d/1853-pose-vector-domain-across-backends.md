### Fixed: every simulation backend validates a scene-placement pose vector

`coerce_pose_vector` documents the invariant "membership, not truthiness" and the
MuJoCo backend has routed `add_object` / `add_robot` / `move_object` /
`add_camera` and `move_to` through it. The Newton and Isaac backends applied it
to `add_camera` only, so their remaining placement methods read the caller's
`position` / `orientation` directly and diverged from that domain in both
directions.

Newton tested the vector for truthiness (`position or [0.0, 0.0, 0.0]`), so a
NumPy array - what pose arithmetic produces, and what the `Args` advertise -
raised a bare `ValueError: truth value of an array with more than one element is
ambiguous` straight through the structured envelope those methods document as
their only failure channel, while an empty vector read as *omitted* and the
object was placed at the default pose under a success result. Everything else
was stored verbatim: a wrong-length position, a `nan`/`inf` component, a `bool`
read as the coordinate `1.0`, a 3-component quaternion, and a bare string stored
AS the position. Isaac coerced with `list(position)`, which validated nothing
either and split a string per character, and its `move_object` /
`set_robot_pose` additionally sliced with `position[:3]` / `orientation[:4]`, so
a 5-component request was written as its first 3.

All seven remaining methods now route both vectors through the shared helper, so
a pose one backend refuses is refused by all of them with the same message, and
a NumPy pose is accepted and normalized everywhere. A new structural test
requires every public backend engine method taking a `position` / `orientation`
/ `target` parameter to use the shared domain, so a new placement method cannot
ship without it.
