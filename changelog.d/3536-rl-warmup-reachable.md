### Fixed: an RL run whose budget never reaches `learning_starts` is refused

The two off-policy RL backends (`fast_sac`, `fast_td3`) take no gradient step
until the replay buffer holds `learning_starts` transitions, and they collect
`max(1, total_timesteps // steps_per_iter)` iterations of
`steps_per_iter = rollout_steps * num_envs` env steps. Nothing related the budget
to the threshold, so a budget that could never reach it spent the entire run on
the uniform-random warmup and then reported success.

Measured on the MuJoCo reach env with `total_timesteps=40`, `rollout_steps=10`,
`learning_starts=64`, `batch_size=16` (so the existing
`learning_starts >= batch_size` relation holds and reports nothing): `validate()`
returned `[]`, `update()` was called **zero** times, and `train()` wrote a
checkpoint, exported it and returned `status="success"` with
`"4 iterations x 10 steps complete"` - while the exported `policy.pt` was
**bit-identical** to the weights `setup()` had just initialized. So the loadable
artifact handed back, the one `create_policy(...)` consumes, was the
initialization rather than a trained policy, and `metrics` carried no
`latest_loss` for the "RUNNING != learning" verdict to read. `fast_td3` behaved
identically.

That is the outcome `rl_replay_problems` already refuses for `gradient_steps=0` -
its docstring records the same "zero gradient updates, yet the run reported
success" - and that the `learning_starts` count domain closes for a value that is
not a count. This was the third route to it, the one where every operand is a
usable count and only the relation between them is wrong.

`validate()` now reports it on both off-policy backends, through a shared
`rl_warmup_reachable_problems` gate rather than a copy per backend, and names two
routes out that were each driven to a gradient step before being named: raise
`total_timesteps` to the next multiple of `steps_per_iter` at or above
`learning_starts`, or lower `learning_starts` to at most what the budget collects.
Naming `learning_starts` itself would not have been a remedy - 64 at 10 per
iteration still collects only 60.

The relation is asked of the loop's own arithmetic rather than of
`total_timesteps` alone, because the floor division is what is collected: 45 steps
at 10 per iteration is four iterations of ten, so a `learning_starts` of 45 is out
of reach even though the budget is not below it, and a bare
`total_timesteps >= learning_starts` test would have admitted it. Every operand is
asked of the shared count domain before the relation is formed, so a non-finite
value is reported once by the domain that owns it rather than deciding the
relation silently. On-policy PPO reads no warmup and stays silent. The shipped
defaults (99984 steps collected against a warmup of 1000) are unaffected.
