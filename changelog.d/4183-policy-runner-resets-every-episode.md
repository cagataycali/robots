### Fixed: a multi-episode policy run resets the policy at every episode, not only when seeded

`run_policy(n_episodes=N)` reset the simulator at each episode boundary but
called `policy.reset()` only inside the seeded branch. A provider that keeps an
observation history or an action queue therefore started episode N conditioned
on episode N-1. On the SO-101 with `flux3_action` an unseeded 10-episode
recording drifted the `shoulder_lift` command by a further ~5 rad per episode
until 4300 of 4500 ticks were clamped at the joint limit, while the recorded
state looked sane. The reset now runs unconditionally, with `seed=None` when
none was given.
