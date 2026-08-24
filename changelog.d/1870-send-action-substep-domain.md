### Fixed: `send_action` refuses a substep count it cannot advance, before it writes

`send_action(action, robot_name, n_substeps)` is the second public stepping
surface and the last one with an unvalidated count. It was left out of the `step`
change deliberately, because it has a contract of its own: it *writes an actuator
target and then advances*, so a count it cannot honor is not merely a bad number
of steps - a refusal arriving after the write leaves the robot commanded and the
world un-advanced.

No backend validated it, and the three did not agree on what a `0` meant.
`n_substeps=0` ran one `mj_step` on MuJoCo and one control step on Newton, and
advanced **nothing** on Isaac, which has no floor at all. So the same call
already meant two different things depending on the backend.

MuJoCo's counter desynchronized from the physics it ran. The loop is
`range(max(1, n_substeps))` and the counter is `step_count += n_substeps`, so the
floored loop and the raw counter disagreed for every count below 1: a `0` ran one
step and recorded none, and a `-5` ran one step and moved `step_count`
**backwards**. A `nan` ran one step and left `step_count` as `nan` - not one bad
call, but every later reader of that world's step count. On Newton's solver-free
path the same `max(1, ...)` added a `2.7` or an `inf` straight into `step_count`.

`3.0` - an integral float, the most innocuous value in the set, and what
`duration / dt` or a config read produces - raised `TypeError` out of `range()`
on MuJoCo and Isaac, *after* the target was written and straight past the
documented `{status, content}` envelope. Its sibling `step(3.0)` accepts the same
number and advances 3, so the two surfaces disagreed about one value. `"3"`,
`None` and `[3]` raised the same way on all three.

All three backends now apply the shared
`positive_whole_number_error` domain before any target is written: the same
scalar policy as `step`'s `non_negative_whole_number_error` with the floor moved
to `1`. A NumPy or integral-float count is honored and coerced, so `3.0` now
succeeds where it used to raise; a fractional, zero, negative, non-finite,
boolean or non-numeric count is refused in the same words on every backend, and
nothing is written when it is.

**`0` is refused rather than honored as "write but do not advance"**, with
`step` named as the surface that advances a count of its own. That was the one
genuine decision here, and it is settled on evidence rather than taste: both
producers of this count already refuse anything below 1 - `PolicyRunner._control_substeps`
returns `>= 1` and raises otherwise, with a docstring recording that clamping `0`
to 1 reinstated the exact under-integration it exists to prevent, and
`training.rl.env.SimEnv` refuses an `n_substeps` below 1. `send_action` was the
only member of that chain without the guarantee. Honoring `0` would instead have
adopted Isaac's *absence* of a floor over the reference backend's explicit one.

`positive_count_error` was the other candidate domain and is wrong here: it
admits only a true `int`, so it would refuse `3.0`, `np.int64(3)` and
`np.uint8(2)` - counts MuJoCo honors today and `step` honors by documented
design. That choice is pinned as a measurement rather than left as a preference.

The `max(1, ...)` floors in MuJoCo's `_apply_sim_action` and Newton's `_advance`
are retained as defensive no-ops and are now unreachable from either public
surface. The counter desynchronization needed no separate fix: it was only
reachable with a value outside the domain.

Out of scope and pinned as such: `send_action` still has **no per-call ceiling**
on any backend, so a count `step` refuses with `_MAX_STEPS_PER_CALL` is accepted
here. That is the resource policy tracked in #1871 - a decision about a number
for three backends with different per-step costs - not an input domain.

Closes #1870.
