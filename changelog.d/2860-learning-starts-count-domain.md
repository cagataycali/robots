### Fixed: a `learning_starts` that is not a count is refused instead of silently taking zero gradient steps

`learning_starts` is one side of the relation `learning_starts >= batch_size`, and
that relation was the only thing standing between the field and its two consumers.
The relation's *other* operand was already asked of `positive_count_error` before
being compared; this one was not, on the stated grounds that it is "one side of a
relation rather than a bare count". That reason is about the shape of the *rule*
and says nothing about whether the value is a count at all, and a comparison is
not a domain: nothing is greater than `nan` and `inf` is below no integer, so a
non-finite value made `learning_starts < batch_size` answer `False`, the relation
passed, and `validate` reported nothing.

Both consumers then read a value that is not a count. `collect_rollout` tests
`buffer.size < learning_starts` to decide whether to draw a random warmup action,
and `train` tests `buffer.size >= learning_starts` to decide whether `update()`
runs at all - so a threshold no buffer can pass disables the whole learning half
of the loop. Measured on the MuJoCo reach env used by
`tests/training/test_rl_fast_sac.py` (40 timesteps, `rollout_steps=10`,
`batch_size=16`, `gradient_steps=2`, one field mutated): `learning_starts=16` gave
`validate() == []`, `train()` success and **3** `update()` calls, while
`float("nan")` and `float("inf")` each gave `validate() == []`, `train()` success
and **0** `update()` calls. The run built the environment, the networks, the
optimizers and the replay buffer, collected every rollout, wrote a loadable
checkpoint and reported `status="success"` having taken no gradient step at all.
That is the outcome `_rl_replay_problems` exists to refuse for `buffer_size` - its
own docstring records the same "zero gradient updates, yet the run reported
success" - reached through the field its scope line excluded. `nan` additionally
skips the random warmup the field exists to provide, so the untrained actor drives
from the first step.

Both operands of the relation are now asked of the same strict-`int` domain, and
the relation only of two values that are counts. The relation is preserved rather
than replaced: a real count below `batch_size` still gets the "must be >=
batch_size" message, so a value that is merely too small is not reported as a
non-count. A very large `int` is still accepted, because magnitude is not the axis
- `10**400` is a count, and a run that has not reached it has genuinely not
finished warming up. Strict does newly refuse an integral float such as `1000.0`
and a `numpy` integer, both of which the bare comparison admitted; the field is
annotated `int`, a relation between a strict count and a loose one is the drift
this closes, and both are pinned as deliberate consequences.

The field-scoped gate's own scope is unchanged: `_rl_replay_problems` still reports
nothing about `learning_starts`, because PPO reads neither it nor the three replay
counts, and the domain is asked in FastSAC's own `validate` beside the relation it
guards. The two claims in `tests/training/test_rl_replay_domain.py` that read
otherwise are corrected in place rather than removed - `tau` remains outside the
count domain entirely, as a coefficient in `(0, 1]`.
