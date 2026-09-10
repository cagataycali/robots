### Removed: `strands_robots.training.reward` (`compute_rabc_weights`, `load_reward_model`, `reward_progress`)

The three helpers were thin wrappers over lerobot's own
`lerobot.rewards.sarm.compute_rabc_weights` script and
`lerobot.rewards.make_reward_model`; nothing in the package, the tools or the
examples called them. The RA-BC loop in `docs/training/overview.md` now shows
the lerobot command for step 2 and points reward-model scoring at lerobot
directly. Training a reward model through `TrainSpec(extra={"reward_model": …})`
is unchanged.
