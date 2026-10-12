### Tests: the learning rate and the RL device join the one-owner table

`_learning_rate_problems` and `_spec_device_problems` are now rows of
`tests/training/test_every_shared_domain_has_one_owner.py`, so every module that
reads `learning_rate` or `device` must route it through the gate, and a local
comparison of either field against a literal is reported. The device gate had no
owner scan before. `tests/training/test_learning_rate_domain.py` goes: its owner
scan is the row, and its other six cells asserted what `torch.optim.Adam` and
lerobot's `AdamWConfig` do, not anything in this package. 76 cells become 71.
