### Added: `FastTd3Trainer` - Twin Delayed DDPG as a native off-policy RL provider

`create_trainer("fast_td3")` now resolves, closing the deliverable epic #858
marked complete without code (#2720, #862). The trainer is the
deterministic-actor peer of `fast_sac`: a tanh-bounded deterministic MLP actor
against twin Q critics with clipped double-Q targets, target policy smoothing
(clipped Gaussian noise on the target policy's action), delayed actor /
target-network updates, and Polyak averaging - the standard Fujimoto et al.
formulation, benchmarked against the FastTD3 project (arXiv:2505.22642) and
re-homed onto the `SimEnv` reward-term backend. It reuses `SimpleReplayBuffer`
and `EmpiricalNormalization`, keeps the `policy.pt` + `policy_meta.json`
checkpoint contract (with `action_keys` from `robot_action_keys`), stores the
*terminal* flag rather than the raw done so time-outs bootstrap, and warms up
with uniform random actions before `learning_starts`.

Unlike the single-env `fast_sac`, collection is vectorized: `num_envs > 1`
steps N independent envs through `VecSimEnv` and pushes N transitions per
tick, storing a done env's next-observation from the captured pre-reset
`infos[i]["terminal_obs"]` - the same capture the vectorized PPO path
bootstraps its next-value from - so the TD target of an episode's last action
is never backed by a fresh episode's first state. Off-policy replay has no
rollout tensor to reshape, so the buffer absorbs the extra envs without
changing the update; `fast_sac` itself remains single-env and unchanged.

The four TD3 knobs land as first-class `RLTrainSpec` fields, each behind a new
shared field-scoped gate under the usual biconditional (a backend that reads
the field routes it through the gate; one that does not stays silent):
`policy_delay` through the positive-count domain - it is the modulus of the
one test that decides whether the actor moves, and a `nan` modulus trains the
critics for the whole run while the deployable actor never takes a gradient
step, under `status="success"` - and `exploration_noise_std` /
`target_noise_std` / `target_noise_clip` through the positive-finite domain,
because zero silently removes the mechanism (a collection that never explores;
plain clipped double-Q reported as the smoothed algorithm), a negative scale
is silently the identical symmetric distribution, and a non-finite one poisons
the actions or the TD target while the run keeps stepping.
