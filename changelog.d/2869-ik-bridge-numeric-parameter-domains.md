### Fixed: an IK knob that cannot be honored is refused instead of solved with

`MinkIKBridge` is the one home for the mink differential-IK solve, re-exported as
the public `MinkIKBridge` of the `cosmos3` and `vera` providers and documented as
a constructor a caller builds directly. It validated two of its arguments
thoroughly - `commanded_dofs` per element, with `bool` rejected by name and every
index range-checked, and `solver` through `resolve_qp_solver` - and handed its
eight numeric knobs straight through to mink.

Every one of those knobs is *applied* rather than forwarded: `max_iters` bounds
the `range` the solve loop iterates, `dt` integrates the joint velocity, `damping`
and the three task costs weight the QP, and the two thresholds decide when the
loop breaks. So an unusable value produced a plausible-looking joint
configuration rather than an error. Measured on a Franka Panda asked to move its
hand 80 mm, a move whose converged residual is 0.761 mm: `max_iters=0` (also
`False`, also a negative count) made `range` empty, so `solve` ran the solver
**zero times** and returned `q_init` unchanged with the whole 80 mm still to go -
a solve that never happened, handed back as one. `max_iters=True` ran exactly one
iteration (14.147 mm). Both convergence thresholds infinite made the *first*
iteration count as converged, byte-identical to `max_iters=1`. A `dt` of `0.0`,
`nan` or `inf`, a `damping` of `nan`, and any infinite task cost each returned an
all-NaN configuration; `damping=inf` produced no joint motion at all.

The remaining spellings failed *late* instead: a fractional, whole-float,
non-finite, string or `None` `max_iters` raised `TypeError` from `range` inside
`solve`, and a negative `damping` raised `qpsolvers`' "matrix P is not positive
definite" - all after the QP backend was resolved, both mink tasks were built and
the bridge had logged that it was ready.

Each knob now reaches the domain its consumer implies, ahead of every side
effect. `max_iters` takes `positive_count_error`, the strict-integer domain, which
is what `range` accepts - the looser whole-number sibling admits `20.0`, and
`range(20.0)` raises. `dt` takes `positive_finite_number_error`. `damping` takes a
local helper composing the shared finiteness domain with the QP's own `>= 0`
floor, so `0.0` stays legal as the undamped solve. The three task costs take
`finite_number_error` only, because mink already refuses a negative cost by name
at task construction and `orientation_cost=0.0` is the documented position-only
solve for arms with fewer than six DOF. The two convergence thresholds likewise
take finiteness only: a threshold no residual can reach means "never break
early", which runs the full budget and is the idiom both solve-loop suites use to
exercise it, so only an infinite one is refused.
