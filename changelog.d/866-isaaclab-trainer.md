### Added: an `isaaclab` trainer - GPU-parallel RL through a separate Isaac Lab install

`create_trainer("isaaclab")` and `train_policy(provider="isaaclab")` launch
`python -m isaaclab train` (rsl_rl PPO) in the interpreter `ISAACLAB_PYTHON`
names, detached, and answer `status(job_id)` from the run's log and exit code:
`latest_iteration`, first/latest/best mean reward, `learning`, `steps_per_s`,
`total_steps`, `training_time_s`, and the rsl_rl run directory as
`checkpoint_dir`. Isaac Lab is never imported and adds no dependency - it pins
its own torch/warp/newton and Kit hard-exits the process that closes it - and
the interpreter comes from the operator's environment, never from the agent's
spec. `validate` refuses a missing interpreter, an unaccepted Omniverse EULA
(`OMNI_KIT_ACCEPT_EULA=YES` is the operator's to set), an unknown `extra` key,
`resume=True` and any `rl_library` but `rsl_rl` before anything launches;
`extra['timeout_s']` stops the run's process group. `Trainer.requires_dataset`
(default `True`) lets `train_policy` reach a provider that reads no dataset, and
the `status` action's JSON block now carries `checkpoint_dir` and
`exported_model` like `train`'s. On one L40S, Isaac-Cartpole with 4096 envs ran
at about 290k environment steps per second.
