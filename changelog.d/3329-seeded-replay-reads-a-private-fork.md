### Tests: the seeded-replay measurement no longer reads the process-global RNG

`tests/simulation/test_rollout_seed_is_applied_or_refused.py` measures the
applied half of the rollout seed contract with a policy whose actions depend on
the seed, and the RNG `set_eval_seed` sets is the process-global one. The policy
drew straight from `random.random()`, so the comparison of two seeded evals also
assumed the rollout was that object's only reader for the length of the rollout.
It is not: any other thread in the interpreter that draws from it between two of
the policy's queries shifts every action after it, and the comparison reports
that as an unapplied seed. The test failed that way once on a branch whose diff
touched no RNG code, taking a required check with it.

Copying the global state into a private generator after `set_eval_seed` was
measured and is not enough: `set_eval_seed` continues into NumPy and torch after
`random.seed`, so the copy is taken hundreds of microseconds later at best, and a
background reader on a 2 ms period still failed the file 1 run in 3. The policy
now reads the global object zero times. Every seeded rollout surface forwards the
seed it applied to `policy.reset(seed=...)` - the contract a service-mode policy
relies on - and the test policy seeds a private `random.Random` from that value,
exactly as such a policy would. Under the same 2 ms reader the file is green
across repeated runs.

Two cases pin it. One run of a seeded pair has a stray draw from the global RNG
in one of two places - a `success_fn` drawing mid rollout, or a NumPy reseed that
also draws, which sits inside `set_eval_seed` after `random.seed` - and the two
runs must still replay identically while deriving identical episode seeds; the
first placement refuses the direct reader, the second refuses the state copy.
And because the policy no longer observes the per-episode reseed, that reseed is
counted as the call it is: every seed `policy.reset` receives was applied with
`set_eval_seed`, in order, so deleting the per-episode reseed from `evaluate`
fails that cell and nothing else.

The unseeded case is measured at the appliers instead of at the state they write.
`assert random.getstate() != expected` could not see the side effect it guarded -
reseeding from entropy and merely consuming draws both leave a state that differs
from the one before the call, so it passed either way. A reseed is a call, and
`set_eval_seed` is the only thing on this path that makes one, so the call is
what is counted.
