### Fixed: the RL deploy fences run the checkpoint on the robot it was trained on

The `Deploying the checkpoint` fence on the reinforcement-learning page and the opening fence of the RL checkpoint policy page trained on `so100` through `make_env` and then rolled the checkpoint out on `so101`, whose joints are named `1`..`6`, so `run_policy` was refused with `observation omits actor_obs_keys the ppo checkpoint was trained on`. Both fences now construct and name `so100`, and a test reads the trained robot off `make_env` and grades every RL deploy fence against it.
