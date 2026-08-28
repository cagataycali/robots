### Fixed: a non-finite IK pose or seed is refused instead of solved with

`MinkIKBridge` now holds all eight of its numeric *knobs* to a shared value
domain in `__init__` - the three task costs and the two convergence thresholds to
`finite_number_error`, `damping` to a local wrapper over it, `max_iters` to
`positive_count_error` and `dt` to `positive_finite_number_error`. The two
*arrays* `solve` reads went straight through, and `solve_trajectory` checked the
one thing a wrong-shaped batch would break on, `poses.shape`, and none of the
values inside it.

Both arrays are applied rather than forwarded: `target_pose` becomes the frame
task's target and `q_init` becomes the configuration the QP warm-starts from.
Measured against a Franka Panda (`nq=9`) on a reachable 80 mm target whose
healthy solve returns nine finite joints, a single `nan` or `inf` anywhere in
either array returned **9 of 9** joints non-finite - as a successful return,
shaped exactly like a converged solve, with nothing in the value or the type to
distinguish it from one. `solve_trajectory([good, bad])` returned **9 of 18**:
the first waypoint a real configuration and the second entirely NaN, so a caller
iterating waypoints receives a partially valid trajectory rather than an error.
One spelling did not come back at all: an `inf` in `q_init` left the QP backend
unable to solve, raising `mink.exceptions.NoSolutionFound` out of a third-party
module rather than the `ValueError` the method documents. The same class of bad
value therefore had two exits, one silent and one naming neither the method nor
the parameter.

Both arrays now reach `finite_vector_error`, the same shared domain the
scene-construction vectors use, checked before the configuration is updated and
before the frame target is set so a refused solve mutates nothing. The pose is
flattened for the check because that domain reads a 2-D argument's *rows* as its
elements and would otherwise refuse a clean `(4, 4)`. One guard in `solve` covers
`solve_trajectory`, which calls it per waypoint; placing the check in
`solve_trajectory` instead would leave a direct `solve` caller unguarded. The
two checks together cost 45.8 us against a 7.6 ms solve on that same Panda, 0.60%
of one call, so there is no budget argument for leaving them to the consumer.

Two things are deliberately unchanged. The bridge never carried the damage across
calls - `solve` re-seeds the configuration from `q_init` every time, so the next
healthy solve was already clean; the guard's placement ahead of the mutation is
craft rather than a repair, and a regression cell pins that a solve after a
refusal is byte-identical to one before it. And `tracking_error` is untouched: it
returns `{"mean_mm": nan, "max_mm": nan}`, a visibly non-finite reading rather
than a plausible one.
