### Fixed: a GR00T fine-tune trains the components the spec asked for, through lerobot

lerobot ships GR00T N1.7 as a native policy, and `LerobotTrainer` has listed
`groot` as a policy type all along - but `build_config` read neither
`TrainSpec.tune` nor `TrainSpec.embodiment`, so a caller who asked to unfreeze the
language backbone got a frozen one and a caller who named an embodiment trained
under the default tag, both while `validate()` reported no problem. Each component
now lands on the field lerobot declares for it (`tune_llm`, `tune_visual`,
`tune_projector`, `tune_diffusion_model`, and the pi0 family's
`train_expert_only`), the embodiment tag lands on `embodiment_tag`, and a
component the resolved policy does NOT declare is refused rather than dropped -
the harm `_policy_supports_expert_only` already documented for the one component
reachable through `method`. A base model likewise goes to `base_model_path` when
the policy declares that field, because `--policy.path` can only read a directory
whose `config.json` names a policy type.
