### Fixed: `set_eval_seed(None)` is refused instead of half-reseeding the process from entropy

`set_eval_seed` is public API, and its shared domain
(`randomization_seed_error`) accepts `None` because for `randomize(seed=None)`
that legitimately means "draw fresh entropy". The applier has no seed to apply
for it, and the three RNGs it drives disagree: `random.seed(None)` and
`numpy.random.seed(None)` reseed from entropy, then `torch.manual_seed(None)`
raised a bare `TypeError` naming neither the parameter nor the method - leaving
`random` and NumPy already reseeded by a call that failed. On an install without
torch the same call reported nothing at all and reseeded both from entropy, so
the two installs disagreed about whether anything had happened.

Either way an unseeded rollout acquired a process-wide RNG side effect it never
asked for, which inverts the rule `PolicyRunner.evaluate` states one layer up:
"a `None` seed leaves the master RNG unbuilt rather than seeding it from
entropy".

`randomization_seed_error` now takes `allow_none`, the same per-destination shape
as the existing `max_seed` ceiling: `None` stays valid where it selects fresh
entropy or means "do not seed", and the applier opts out with `allow_none=False`.
The refusal names the parameter and the reason, sits ahead of every RNG so it has
no side effect either, and the messages for other unusable values stop
advertising `None` at a destination that refuses it. Callers that want the RNGs
untouched should not call `set_eval_seed` - which is what every caller in the
module already does.
