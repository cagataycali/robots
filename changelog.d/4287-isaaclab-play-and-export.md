### Added: an Isaac Lab run can be played back on video and deployed with `create_policy("rl")`

A policy trained with `train_policy(provider="isaaclab")` could be trained and
polled, and nothing else: seeing it move meant running `isaaclab play` by hand
with the right physics preset, and `create_policy("rl")` could not load it,
because rsl_rl's actor (an MLP with the run's own activation - ELU in the Isaac
Lab task configs - and an optional observation normalizer) is not strands'
Tanh PPO network.

`train_policy(action="play", provider="isaaclab", job_id=...)` plays the run's
newest checkpoint in Isaac Lab with the task and physics preset it trained on
(from `strands_run.json`), records a clip with the Kit visualizer, and reports
it in `metrics["video"]`; Isaac Lab's TorchScript export of the actor is
`exported_model`. `train_policy(action="export", provider="isaaclab")` converts
the newest `model_<iteration>.pt` into `strands_policy/policy.pt` +
`policy_meta.json` with `provider="rsl_rl"`, which
`create_policy("rl", checkpoint_dir=...)` loads without rsl_rl or Isaac Lab
installed. The actor reads Isaac Lab's concatenated observation group as
`policy_obs.<i>`, or whole as a `policy_obs` vector. On the twelve policies
trained on one L40S, the converted actors match Isaac Lab's own TorchScript
export exactly.
