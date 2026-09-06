### Fixed: the interval an RL run checkpoints at is held to one shared domain

`RLTrainSpec.log_interval` was documented as "iterations between progress logs",
but no RL module emits a log line: the field is read in exactly one expression,
and that expression decides whether `save_checkpoint` runs for the iteration. It
is the RL run's checkpoint cadence - the same kind of value as
`TrainSpec.save_freq`, consumed as the modulus of the same periodic-checkpoint
test - and it reached that modulus with no domain, so a cadence refused for a
supervised run was accepted for an RL one.

Measured on the inherited loop over 20 iterations, against the
`[1, 6, 11, 16, 20]` an `int` cadence of `5` writes: `True` wrote 20 checkpoints
(a modulus of one), `2.5` wrote the schedule of `5`, `nan` wrote only the final
one - silently the *disabled* mode, under `status="success"` - and `"5"` raised
`TypeError` out of `train()` after `setup` had built the env, the networks and
the optimizers, past a `validate` documented to return every problem a run has.
For RL those intermediate checkpoints are not a convenience: return is
non-monotonic in training, so the deployable policy is often an earlier
iteration.

All three RL backends now route the field through the shared
`step_cadence_error` domain that already owns `save_freq`. Only the *type* is
graded, so `0` remains the supported "no intermediate checkpoints" mode, and the
spec entry now documents what the field paces.
