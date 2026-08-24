### Fixed: the reproducibility seed a `TrainSpec` asks for is one shared non-negative-count domain

`TrainSpec.seed` is read by four backends and none of them checked it, while its
appliers disagree about what a single value means. `torch.manual_seed` -- the
applier on both RL trainers -- reduces its argument modulo `2**64`, so `seed=-1`
was *silently* `seed=2**64 - 1`: the run was reproducible under a number nobody
asked for, and collided with a seed another caller could legitimately have named
(`manual_seed(-1)` and `manual_seed(2**64 - 1)` draw the identical stream, as do
`True`/`1` and `2.7`/`2`). On the LeRobot path the same value reached lerobot's
`set_seed`, which reseeds Python's `random` and *then* hands it to NumPy, which
refuses a negative -- so a rejected seed left the process RNG reseeded by a call
that failed, under NumPy's message rather than one naming the field. Cosmos and
LeRobot's argv-parity path rendered every value, including `nan` and `[7]`, into
an override or a flag token that failed inside the run after the dataset and
model were loaded, if at all.

`Trainer._seed_problems` now checks the field against the one shared
`non_negative_count_error` domain -- the same non-negative-integer rule, whose
`0` is first-class here too because seed `0` is a seed -- and the four backends
that read the field call it. `None` remains the documented "use the backend's own
default" sentinel, and the two backends that never read the field (`mock`,
`groot`) do not report on it.
