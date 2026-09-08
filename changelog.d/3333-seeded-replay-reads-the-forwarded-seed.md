### Tests: two more seeded-replay cells no longer compare the process-global RNG across a rollout

A rollout seed's reproducibility half is only observable through something whose
output depends on the RNG state the seed sets, and the RNG a rollout seed sets is
the *process-global* one. Two cells measured it by reading that shared object for
the length of a rollout, so each also assumed the rollout was its only reader:
`test_run_policy_seed_reproducibility.py::test_same_seed_same_trajectory` drew
from `random.random()` / `np.random.random()` inside the policy double, and
`test_policy_runner_benchmark.py::TestEvalSeeding::test_evaluate_benchmark_reseeds_per_episode`
drew from the `random` module inside the benchmark spec's `on_episode_start`. One
stray draw between two of those reads shifts everything after it, and the
comparison reports it as an unapplied seed - a failure for a draw the rollout
never made.

Each cell now reads the seeded value the rollout hands it and the shared object
zero times: the policy double seeds a private `random.Random(seed)` /
`np.random.default_rng(seed)` in `reset(seed=...)`, and the benchmark spec draws
from the `episode_rng` the eval loop derives for that episode. What they stop
observing - that the rollout reseeds the global RNGs at all - is a call, and two
new cells count it as one, on the single-rollout path and per episode.

`TestEvalSeeding`'s two direct `set_eval_seed` draw-comparisons are removed: a
cell whose subject is a global reseed has to read the global object, and that
contract is already pinned, over a wider set of seeds, in
`test_set_eval_seed_requires_a_seed.py::TestUsableSeedsAreStillApplied`. Pinning
it in one place with the shortest window, rather than in two, is the whole
remedy available to that shape.

A tree-wide check keeps the property: a `Policy` subclass in the suite may build
a private generator, and may not draw from the shared one.
