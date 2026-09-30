### Added: the `isaaclab` trainer trains with skrl (AMP humanoids, multi-agent MAPPO/IPPO)

`extra['rl_library']` accepted only `rsl_rl`, so every Isaac Lab task whose agent
configs are skrl's could not be trained through strands: the AMP humanoids (Walk, Run,
Dance), the multi-agent `skrl_mappo` / `skrl_ippo` configs (Pendulum-MARL,
Shadow-Handover) and the Cartpole showcases. `rl_library="skrl"` now launches `python
-m isaaclab train --rl_library skrl`, names the run after the job with
`agent.agent.experiment.experiment_name` (skrl has no `--run_name`), finds its
`checkpoints/agent_<timestep>.pt`, and reads the mean reward from the run's TensorBoard
events (skrl prints none) with a dependency-free reader, reporting the same metrics as
an rsl_rl run. `learning_rate` lands on `agent.agent.learning_rate` with the KL-adaptive
scheduler pinned off; `save_freq` (rsl_rl iterations) is refused for skrl, and `export`
names a skrl checkpoint as not yet convertible. On one L40S: Cartpole 50 iterations
reward -0.32 -> 4.83, AMP-Walk 20 iterations, Pendulum-MARL MAPPO 0.35 -> 4.00.
