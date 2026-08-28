### Fixed: a non-finite IK configuration is refused instead of reporting a pose for it

`MinkIKBridge` has exactly two methods that *apply* a joint configuration to its
own state - both call `self._configuration.update(...)` - and only one of them
held that array to a value domain. `solve` gained `finite_vector_error` for its
`q_init`; `ee_pose`, the forward-kinematics reader with 19 call sites across five
modules, read the same kind of array into the same call one line down and checked
nothing.

Measured against a Franka Panda (`nq=9`) whose healthy `ee_pose` returns a pose
with sixteen finite entries, a single `nan` or `inf` anywhere in `qpos` returned a
`(4, 4)` pose with **12 of 16** entries non-finite - as a successful return,
shaped exactly like a reachable pose. Three consumers inherited it. A
`norm(ee[:3, 3] - target)` residual, the shape six call sites across the two
motion-primitive backends use, came back `nan`, and `nan <= threshold` is
`False`, so a convergence test never fired. `tracking_error` reported
`{"mean_mm": nan, "max_mm": nan}`. And the closed loops in the Cosmos3 and VERA
IK bridges compose a pose delta onto this pose and solve for the result, so
`solve`'s own guard refused one step later naming `target_pose` - an argument the
caller never supplied. The caller supplied the seed.

That last one is why this wants a guard rather than a docstring: guarding `solve`
turned a silent wrong answer into a *misattributed* refusal, and only a check at
the method reading the caller's own array can name it. `qpos` now reaches
`finite_vector_error`, the same shared domain `solve` uses, checked before the
configuration is updated so a refused call mutates nothing. The check costs
8.466 us against a 3759 us solve, and every in-package consumer pairs `ee_pose`
with a solve or a norm, so it is 0.224% of a closed-loop step; against a bare
`ee_pose` of 21.66 us it is 64.2%, which is the honest figure for
`tracking_error`'s per-row loop and about 8 ms on a 1000-step trajectory.

The domain is called verbatim rather than fronted by a cheaper
`np.all(np.isfinite(...))` test, because that test *accepts* a 0-d scalar and a
2-D column which the domain refuses - so a hand-rolled fast path would decline to
judge two spellings its own sibling rejects. Both of those previously escaped as
an `IndexError` or as `ValueError: could not broadcast input array from shape
(3,1) into shape (3,)`, naming neither the method nor the parameter.

Two things are deliberately not claimed. The bridge never carried the damage
across calls - `solve` re-seeds the configuration every time, so the next healthy
call was already clean; the guard's placement ahead of the mutation is craft
rather than a repair, and a regression cell pins that the configuration is
untouched by a refusal. And `tracking_error` gains its refusal by *inheritance*
rather than by a guard of its own: it was left alone when `solve` was guarded
because a `nan` reading is visibly non-finite, and this change reaches it through
the method it calls per row, so a non-finite solved trajectory is now named
rather than reported.
