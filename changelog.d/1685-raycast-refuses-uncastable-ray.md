### Fixed: a ray the caster cannot cast is refused, not reported as a miss

`raycast` and `multi_raycast` answer clearance and obstacle questions, so "no
intersection" is a load-bearing answer. Two inputs produced that answer without
casting anything.

`multi_raycast` validated each direction inside its cast loop and, for a
malformed one, appended `{"distance": None, "geom_id": None, "error": ...}` and
carried on. `distance: None` is exactly what a genuine miss reports, the overall
`status` stayed `"success"`, and the summary folded the rejected ray into the hit
denominator (`"Multi-ray: 4/8 hits"`), so a bearing that was never cast read as
free space - on a fan whose bearing 3 lost a component, the obstacle 0.741 m
ahead of it was reported as clear. The batch parameter itself was unguarded too:
a bare string was iterated one ray per character, a non-sequence raised
`TypeError` past the tool-error contract, and an empty batch reported `0/0 hits`.
Every direction is now validated before any ray is cast, and a batch holding one
it cannot cast is refused with every offending index named - matching `raycast`,
which already refused the same directions outright.

`exclude_body` reached `mj_ray` unchecked on both methods. A fractional / string
/ `nan` value raised `TypeError` out of the pybind11 signature, and an id outside
`[0, model.nbody)` matched no body, so the geoms the caller asked the ray to pass
through were silently included and could be reported as the obstacle. Both
methods now accept only `-1` (exclude nothing) or an id the compiled model
defines, rejecting `bool` explicitly.
