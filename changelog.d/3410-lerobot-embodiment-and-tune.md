### Fixed: a lerobot GR00T fine-tune trains the components it was asked to

`LerobotTrainer` lists `groot` among the policy types it trains, and lerobot's
own `GrootConfig` declares an `embodiment_tag` plus four per-component switches
(`tune_llm` / `tune_visual` / `tune_projector` / `tune_diffusion_model`). Two
`TrainSpec` fields are documented as exactly those requests, and the trainer
read neither: `spec.embodiment` was read nowhere in the module and `spec.tune`
only for the lora/expert-only mutual-exclusion check. So
`TrainSpec(embodiment="so100_arm", tune={"llm": True, "visual": True,
"projector": False, "diffusion": False})` built a config carrying
`embodiment_tag="new_embodiment"` and the inverse of the request on all four
toggles - a full fine-tune of the wrong parameters against the wrong head, with
`validate()` reporting no problem.

Both fields now reach the policy config, and a policy whose config lacks them
refuses the request instead of dropping it - the rule `learning_rate` and
`relative_actions` already follow in the same method. The capability is probed
off lerobot's registered config class (`_policy_supports_embodiment_tag`,
`_policy_tune_components`), beside the three sibling probes, so a policy lerobot
adds with an embodiment tag or component toggles is honoured on arrival. A
`tune` key naming no component (`vision` for `visual`) is refused rather than
silently matching nothing, and `build_command`'s argv carries the same flags in
a canonical order. Each toggle's value is checked as a boolean rather than read
by truthiness: `tune={"llm": "false"}` - the spelling a YAML- or JSON-sourced
config carries - is refused by name instead of training the component the
caller asked to freeze.
