### Fixed: an RL trainer's `train()` closes the env its `setup()` built

The trainer owns the env lifecycle - `RLTrainSpec.env_factory` hands it a
factory precisely so nothing else holds the instance - but no RL `train()`
ever called `env.close()`. For a single `SimEnv` that was invisible (its
`close` is a documented no-op, the engine belongs to the caller), which is why
the leak went unnoticed: a vectorized run builds a `VecSimEnv` whose reused
`ThreadPoolExecutor` is only shut down by `close()`, so every vectorized
`train()` left `min(num_envs, 8)` idle worker threads alive for the rest of
the process while the run reported success.

`BaseRLAlgo.train` (which PPO inherits) and the off-policy overrides (FastSAC,
FastTD3) now close the env in `finally`, so the pool is released whether the
run finished or raised - a raise out of `setup` included, where a partly-built
run could already hold a live pool. The close is safe on every path by
construction of the envs' own `close()`: a validation failure never built an
env, `SimEnv.close` stays a no-op, `VecSimEnv.close` is idempotent, and a
closed `VecSimEnv` still steps and resets serially, so the documented
train-then-`evaluate()` continuation on the same trainer instance keeps
working. `evaluate` itself never closes - it leaves the trainer live for
exactly that reuse - so nothing double-closes.
