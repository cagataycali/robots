### Fixed: the training examples install what LeRobot's `train()` needs

`examples/07_post_tune_any_policy.py` and `examples/17_judge_recorded_episodes.py` documented
`pip install "strands-robots[sim-mujoco,lerobot]"` and then trained through `create_trainer("lerobot_local")`.
LeRobot's `train()` requires `accelerate` on CPU as well as GPU and no strands extra supplies it, so both
examples ran every earlier stage and then exited on a rejected `TrainSpec`. Both install lines now name
`"lerobot[training]"`, and `tests/test_examples_document_the_interface_they_have.py` grades the rule against
the trainer's own roster of call-time packages.
