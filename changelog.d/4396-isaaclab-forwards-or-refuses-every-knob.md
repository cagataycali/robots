### Fixed: the `isaaclab` trainer forwards every TrainSpec field and Isaac Lab knob it is given, or refuses it

`base_model`, `num_gpus`, `num_nodes` and `save_freq` were accepted and silently
dropped - a "fine-tune from model_600.pt" trained from scratch and reported success -
while `resume` was refused although Isaac Lab's `--checkpoint` does it, and `extra`
refused everything outside six keys. Now `base_model` (a `model_<iteration>.pt` or a
run directory) and `resume=True` (the newest run of the task in `output_dir`) become
`--checkpoint`, and the job's iteration count continues from the checkpoint's;
`save_freq` becomes `agent.save_interval` (left at its default, the task keeps its
own); `num_gpus`/`num_nodes` above 1 are refused, naming Isaac Lab's multi-GPU
launch. `extra['overrides']` takes `{"env.<path>" | "agent.<path>": scalar}` Hydra
overrides, each checked against the task's real env and agent configs in the Isaac
Lab interpreter before launch - Hydra itself adds a misspelt `env.` path and trains
the default - with the closest field names in the refusal; `extra['agent']` selects
an agent config entry point (recurrent, symmetry...), and `extra['device']`,
`video` / `video_length` / `video_interval` and `deterministic` reach their flags.
`learning_rate` also pins `agent.algorithm.schedule=fixed` unless an override picks
a schedule: rsl_rl's adaptive schedule, in 54 of 57 Isaac Lab PPO configs, replaced
the rate from the first iteration and capped it at 1e-2. The run record keeps every
override, and `play()` replays the `physics=` and `env.*` ones and the agent config.
