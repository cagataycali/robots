### Fixed: `step` no longer holds the engine lock for the whole step count

`step(n_steps)` held `self._lock` for the entire count on the Isaac and Newton
backends, so every other locked method on the engine blocked for the duration of
the call. Measured solver-free, counting lock acquisitions for one
`step(100_001)`:

| | lock acquisitions | ticks advanced |
|---|---|---|
| MuJoCo | refuses the count (its own ceiling) | 0 |
| Isaac | 1 | 100_001 |
| Newton | 1 | 100_001 control = 400_004 solver steps at `substeps=4` |

At Isaac's ~2 ms `world.step` that single hold is over three minutes with nothing
able to interleave -- and Isaac is where it matters most, because `pump` /
`run_pump_forever` drive the sim from the owning main thread precisely so a web
UI can serve `get_observation` / `send_action` on worker threads. A worker's
locked read is exactly what waited. The same call now takes 102 and 101
acquisitions.

MuJoCo already batched its loop, so `_STEPS_PER_BATCH` moves onto `SimEngine`:
the *reason* for the granularity is shared across backends even though the
per-call ceiling's value is not, which is why the ceiling stays MuJoCo's own.

Copying the batching across was not sufficient, because the batching was itself
unsafe at its boundaries. `cleanup` nulls `self._world` under a bounded acquire
of that same lock so a worker is never inside `mj_step` on freed arrays, and a
batched loop releases the lock every 1000 steps -- so the handoff lands between
two batches, which nothing checked:

```
MuJoCo step(3000), world nulled on the release ending batch 1
  before: AttributeError: 'NoneType' object has no attribute '_model'
          raised past the structured envelope, after 1000 of 3000 ticks
  after : status=error, "step: world was destroyed mid-run after 1000 of 3000
          steps; aborting. The steps already advanced are not rolled back."
```

Every backend now re-checks the world on each batch boundary and aborts through
its documented result channel, naming the steps completed -- some of them were,
and a bare "no world" would read as the call having done nothing. This is the
pairing `_primitive_abort_reason` already made for the motion-primitive loops,
which release the lock on the same schedule and re-check for the same reason.

Note that a release bounds the lock *hold*, not handover latency: Python locks
are unfair, and a thread already blocked on the lock was measured waiting 333 ms
of a 400 ms call. `_STEPS_PER_BATCH` is not a latency guarantee.

The per-call ceiling remains MuJoCo-only and is still tracked as a separate
decision: the batching bounds how long the lock is held, the ceiling bounds how
much work is accepted at all, and one number cannot express one resource policy
across backends whose per-step cost differs by an order of magnitude.
