### Fixed: PPO refuses an optimization-epoch count it cannot honor

`RLTrainSpec.num_learning_epochs` is the loop bound of PPO's entire optimizer
step -- `for _ in range(spec.num_learning_epochs)` encloses every
`optimizer.step()` in the update -- and nothing checked it. A non-positive value
therefore took **no gradient step at all** while the run still collected its
rollouts, wrote a deployable checkpoint and reported `status="success"`; the
metrics read `0.0` rather than blank because the update averages its
accumulators through `max(1, n_updates)`, so an epoch count that ran no
minibatch reported plausible losses for a run that learned nothing. Measured
over a 60-step run, `0` and `-3` both produced 0 optimizer steps and a
checkpoint bit-identical to the untrained initialisation. `True` was a silent
single epoch (12 optimizer steps instead of 24), and a non-integer raised a bare
`TypeError` out of `range()` after the environment, the networks and a full
rollout had already been built.

`Trainer._optimization_epochs_problems` now holds the field to
`positive_count_error`, the domain this repository already uses for a value
consumed as a `range()` bound. It is scoped to the on-policy backend: FastSAC
optimizes per gradient step from a replay buffer and has no epoch loop, so it
must not report on a field it never reads.
