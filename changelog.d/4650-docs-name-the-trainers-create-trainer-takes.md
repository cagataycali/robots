### Docs: the README and "Teach it" name the trainers `create_trainer` takes

The README's Train row named LeRobot, Cosmos 3, PPO and FastSAC as brands only, and
`docs/start/teach-it.md` showed `create_trainer("lerobot", ...)`, which raises
`ValueError`. Both now give the registered names (`lerobot_local`, `cosmos3`, `ppo`,
`fast_sac`, `fast_td3`, `sagemaker`), and a test fails when a page names a trainer
the registry does not have.
