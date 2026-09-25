### Docs: the RL pages document all three trainers, and the environment count is the trainer's

`create_trainer("fast_td3")` resolves `FastTd3Trainer`, the `[rl]` extra table
advertises it and `docs/policies/rl.md` describes "the three backends", but
`docs/training/rl.md` documented two: no component-table row, no section, and no
domain on either page for the four fields only that backend reads
(`policy_delay`, `exploration_noise_std`, `target_noise_std`,
`target_noise_clip`) or for the shared replay counts `gradient_steps` /
`batch_size` / `buffer_size`. The reference table also attributed fields to
fewer trainers than read them - `both` on `total_timesteps`, `rollout_steps` and
`gamma`, which all three read, and `FastSAC` on `tau`, which both off-policy
trainers read - so a caller was sent to the wrong domain or to none.

The single-environment constraint was stated as the MuJoCo backend's; it is
FastSAC's. `fast_td3` at `num_envs=4` and `ppo` at `num_envs=3` both train and
deploy a reach policy on MuJoCo through `VecSimEnv`, while `fast_sac` refuses any
count but `1`. `RLTrainSpec.num_envs`'s own docstring carried the same framing
and is corrected with the pages.
