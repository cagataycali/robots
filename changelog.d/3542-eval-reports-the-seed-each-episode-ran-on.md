### Fixed: an evaluation reports the seed each episode ran on, on both of its routes

`PolicyRunner.evaluate` has two routes - a `spec` (delegated to
`_evaluate_with_spec`) and a `success_fn` - and both draw one per-episode seed
from the same master RNG, reseed the process with it, and forward it to
`policy.reset`. Only the `spec` route reported it, so on the `success_fn` route
the value was drawn, used, and dropped. Measured on an SO-101 in MuJoCo at
`seed=42`, both routes drew the identical seeds
`[478163327, 107420369, 1181241943, 1051802512]`; the spec route handed all four
back as `episodes[i]["seed"]` and the other handed back none.

That is the field a caller needs most on the route that carries no dense reward
and no `info`: the payload says episode 3 of 10 failed, and replaying episode 3
alone requires the seed it ran on. Reconstructing it means replicating a private
master RNG and its draw count. An unseeded eval builds no master RNG at all, so
it now reports `seed: None` rather than a number that replays nothing.

The `Args:` entry for `seed` also said the value was "Only used when `spec` is
provided", describing this route before it was seeded and contradicting
`SimEngine.eval_policy` - the `success_fn`-only facade - which documents that
"each per-episode seed is forwarded to `policy.reset`". The facade was right;
the entry now says so, and states where the per-episode seed is reported.
