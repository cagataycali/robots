### Added: the `isaaclab` trainer trains with rl_games (Factory, Forge, AutoMate)

Eight Isaac Lab tasks register rl_games agent configs only - Factory PegInsert,
GearMesh and NutThread, Forge, AutoMate Assembly and Disassembly - so with rsl_rl and
skrl alone strands could not train them. `rl_library="rl_games"` now launches them,
names the run `<start time>_<job id>` through `agent.params.config.full_experiment_name`,
reads progress from rl_games' `epoch: N/M` lines and the mean reward and task success
rate from its TensorBoard `summaries/`, and finds `nn/last_<cfg>_ep_<epoch>_rew__<r>_.pth`;
`learning_rate` lands on `agent.params.config.learning_rate` (identity schedule) and
`save_freq` on `save_frequency` (epochs). On one L40S, Factory PegInsert trains 5
iterations through `train_policy`, reward 37.9 -> 45.2. The Isaac Lab venv needs
`pip install rl-games gym==0.26.2` (Isaac Lab 3.0rc1 ships no rl-games extra).
