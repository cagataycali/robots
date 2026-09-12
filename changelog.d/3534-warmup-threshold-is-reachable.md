### Fixed: an off-policy run whose warmup threshold is out of reach is refused

`learning_starts` is the replay fill FastSAC's and FastTD3's first gradient step
waits for, and nothing checked that the run it configures can reach it. Two
caller-supplied counts bound the fill a run ever reaches - the step budget it
collects, `max(1, total_timesteps // steps) * steps`, and `buffer_size`, the ring
buffer's capacity - and either one below the threshold takes **zero** gradient
steps for the whole run. Measured on the MuJoCo reach env, on both backends:
`total_timesteps=20` against `learning_starts=32`, and `buffer_size=8` against
`learning_starts=16`, each returned `[]` from `validate()` and then
`status="success"` with a written checkpoint and an exported `policy.pt` - the
randomly initialized network `setup` had built, no gradient step having run.

The existing `learning_starts >= batch_size` relation does not cover this: it
sizes the first batch, not the wait for it. Both cases are reachable with plain
positive `int` values that pass every per-field domain, and `buffer_size=1`
builds the same one-slot buffer as `buffer_size=True`, which the strict-`int`
count domain already refuses on the grounds that it "never reached
`learning_starts`" - the same buffer, the same run, two verdicts.

`validate()` on both off-policy backends now reports each short count on its
own, naming the threshold it cannot reach and the count to raise, so a caller
sees every value it got wrong in one pass. The relation is graded only between
counts: a non-count in any operand is left to the gate that names that field
rather than described as an unreachable threshold, and the reachable domain is
untouched - the shipped defaults, a threshold met exactly, and a budget below
one iteration (which still collects a whole one) all pass.
