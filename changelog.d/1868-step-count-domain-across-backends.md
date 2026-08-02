### Fixed: `step` validates its count on every simulation backend

`step(n_steps)` is the most-called method on the simulation surface and was the
last public numeric input with three different domains. MuJoCo has *documented*
one since its numeric inputs were hardened - "Non-negative step count (`0` is an
accepted no-op)" - but implemented it by hand; Newton and Isaac validated
nothing.

A negative count was a silent success on Isaac. `range(-5)` is empty, so the call
stepped nothing and reported `status="success"` - then divided the elapsed wall
time by that negative count and put the result in agent-visible text:
`Stepped -5x. sim_time=0.0000s, wall=0.0ms, -11876485 steps/sec`. `False` read
the same way, as `Stepped Falsex`.

`step(0)` advanced the world on Newton. `_advance` floors its count at
`max(1, n_steps)`, so a zero - the value MuJoCo documents as a no-op - stepped
once while the result said `Stepped 0 step(s).`: the report and the world
disagreed, and one call was a no-op on one backend and a step on another. `-5`
and `nan` reached that same floor and also advanced one step, reported as
`Stepped -5 step(s).` and `Stepped nan step(s).`. `step` now answers its own zero
rather than moving that floor, because `send_action(n_substeps=)` shares
`_advance` and reads the floor as its own contract.

`inf` escaped MuJoCo's own envelope: `int(float("inf"))` raises `OverflowError`,
which the hand-rolled `except (TypeError, ValueError)` did not catch, so the one
backend that documented this domain raised a bare exception through the
structured result these methods document as their only failure channel. On
Newton's solver-free path the same value left `step_count` and `sim_time` as
`inf` permanently, so every later `get_state()` reported `t=inf`; `2.7` left the
float `2.7` in an integer step counter. A boolean was read as a count of one on
all three backends, and MuJoCo truncated `2.7` to two steps under a success
result while its docstring promised an error for a non-integer.

Two paths reach this from outside the process and now get a structured refusal
instead of a raise or a false success: the mesh command router
(`r.step(cmd.get("steps", 1))`, whose `dict(...)` call site propagates a raise)
and the device-connect `@rpc()` `step`, which forwards a remote caller's value
unchanged. Internal callers were already safe - `PolicyRunner._control_substeps`
returns a true positive `int` on every path.

The shared domain is `non_negative_whole_number_error`, the missing cell of an
existing 2x2: it stands to `positive_whole_number_error` exactly as
`non_negative_count_error` stands to `positive_count_error`, and for the same
recorded reason - its `0` is first-class rather than degenerate. Neither existing
helper fitted. `non_negative_count_error` has the right floor but accepts only a
true `int`, so it would have refused the `3.0` and `np.int64(3)` MuJoCo honors
today (the first is pinned by an existing test), and `positive_whole_number_error`
has the right scalar policy but would have refused the documented `0`. All three
backends now apply it before any lock, solver or stage work and then coerce once
with `int()`, which is safe because the guard has already performed that same
conversion and compared the result back. The refusal is identical word for word
across the three, not merely identical in verdict.

Every real scalar gets a verdict from that guard; nothing raises out of it. That
is the contract rather than a detail, because these methods document their
structured result as the only channel a bad count is reported through and one of
them takes its count from a remote process, where Python integers are
arbitrary-precision. So the integrality test is an `int()` in a `try` rather than
a `float()` round-trip - `float(10**400)` raises `OverflowError`, which would turn
a refusal into a crash for the values most in need of one - and the refusal text
is rendered only when a refusal is actually returned, since `repr` of an `int`
past `sys.get_int_max_str_digits()` (4300 digits) raises `ValueError` and such a
count is *accepted*. Magnitude is not part of this domain: `10**400` is a
non-negative whole number and is accepted as one, and whether a count is too
large to advance in a single call stays the per-call ceiling's question rather
than becoming a silent boundary at the float range.

Two neighbours are deliberately left alone and pinned as out of scope:
`send_action(n_substeps=)`, which has the same gap on a second public surface,
and the per-call ceiling - MuJoCo refuses a count above `_MAX_STEPS_PER_CALL`
(100_000) while Isaac and Newton have no equivalent. That is a resource policy
rather than an input domain, and choosing one ceiling for three backends with
different per-step costs is a decision rather than a defect. (The batched lock
release that was MuJoCo-only at the time of this change is no longer part of that
asymmetry; it was made shared separately.)
