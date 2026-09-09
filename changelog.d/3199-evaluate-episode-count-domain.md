### Fixed: an RL evaluation's episode count is held to the shared count domain

`BaseRLAlgo.evaluate` reads `num_episodes` three times - as the `range()` bound
of the episode loop, as the denominator of the reported `success_rate`, and
verbatim as the `num_episodes` field of the returned metrics dict - and guarded
it with a bare `num_episodes <= 0` test, which screens none of those.

`True` ran one episode and came back as `{"num_episodes": True}` from a field
documented `int`, with the success rate divided by a flag, while `False` was
refused. `2.5` / `3.0` / `nan` / `inf` passed the comparison and raised
`TypeError` out of `range()` - after the env, the networks and the optimizers had
been built and a checkpoint loaded - from a method documenting
`Raises: ValueError`.

That raise also escaped the eval-mode window before it was closed. `evaluate`
flips the actor-critic and the observation normalizer into `eval()` mode, and
`EmpiricalNormalization` freezes its running statistics there. `collect_rollout`
re-enters `actor_critic.train()` itself, but `evaluate` is the only place in the
package that puts the normalizer back, so observation normalization silently
stopped learning for the rest of the run - measured on a MuJoCo SO-101 PPO
trainer, 3 of 7 episode counts left it frozen.

`num_episodes` now consults `positive_count_error`, the same domain this
package's other `range()`-bound counts (`total_timesteps` / `rollout_steps` /
`num_envs`) already use, and the mode restore moved into `finally` so the stated
side-effect-free contract holds on every exit rather than only the happy path - a
caller-supplied reward term or `success_fn` raising on a live engine froze
normalization the same way.
