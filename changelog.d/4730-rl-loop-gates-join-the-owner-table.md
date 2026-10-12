### Tests: the two RL loop gates join the one-owner table

`_rl_run_size_problems` and `_rl_checkpoint_interval_problems` are now rows of
`tests/training/test_every_shared_domain_has_one_owner.py`. A row can name the
module whose inherited loop reads the field (`base_algo.py`), and the backends
that must call the gate are then derived from the class hierarchy, so a new RL
backend that only inherits the loop is still checked. The two per-field suites
drop their private copies of that scan, four cells that asserted Python
arithmetic, and a `num_envs` sweep `test_rl_env_count_domain.py` already runs.
151 cells become 128.
