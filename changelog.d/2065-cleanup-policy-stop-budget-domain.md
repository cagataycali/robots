### Fixed: a `cleanup` stop budget the join cannot measure abandoned a live policy worker

`MuJoCoSimEngine.cleanup` waits on each live policy Future before nulling the
world, because a worker still inside `mj_step` on freed arrays is a
stale-pointer segfault. That wait is
`Future.result(timeout=policy_stop_timeout)`, and the caller-supplied budget
reached it unchecked. `Future.result` measures its wait as
`time.monotonic() + timeout`, so `0`, a negative value and `nan` expired it
before its first check and `inf` - the spelling that reads as "wait as long as
it takes" - raised `OverflowError` out of that arithmetic. Measured against a
live 50 Hz rollout, the documented default waited for the worker to unwind
while `inf`, `nan`, `0` and `-1` each returned in ~11 ms with the worker still
running and the world freed; `True` silently capped a 5 s budget at 1 s. A
non-real budget was worse still: `Future.result` raised `TypeError`, and the
`%.1f` in the join's own warning then raised too, so the record reporting the
skipped wait was dropped and the operator saw nothing at all.

`policy_stop_timeout` is now held to the same positive-finite domain every
other span of time is (`positive_finite_number_error`). A budget outside it is
reported against the parameter it came from and resolved to the documented
default - the same thing `None` already means - rather than refused, because
`cleanup` is the release path that `__exit__` and the finalizer both call:
raising would leak the world, the executor and the renderers for a value error.
The resolved budget is normalized to `float`, which is load-bearing rather than
cosmetic - the shared guard accepts any real scalar and `Future.result` refuses
a `np.float32`.
