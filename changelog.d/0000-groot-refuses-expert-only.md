### Fixed: a GR00T tuning strategy that reaches no flag is refused

`Gr00tTrainer.validate` accepted `method="expert_only"` and forwarded it
nowhere. GR00T freezes `tune_llm` / `tune_visual` / `tune_projector` /
`tune_diffusion_model` individually and has no single expert-only switch, so
`_resolve_tune` -- which branches on `frozen_backbone` -- had no branch for it:
the four flags were byte-identical to a plain `full` fine-tune, and `validate`
reported no problems. A caller who asked to train only the action expert trained
the projector as well and was told the run succeeded. lerobot's own gate already
refuses `method="expert_only"` for its native `groot` policy for the same reason
(`GrootConfig` carries the four switches, not a `train_expert_only` field).

`expert_only` is now refused by name, and the refusal names the component set
that expresses it -- `tune={"projector": False}` trains the diffusion action head
and freezes everything before it -- as well as the lerobot policies whose configs
carry `train_expert_only`. Which components the action expert spans is GR00T's
decomposition to state, so the set is named rather than assumed. `full` and
`frozen_backbone` are unchanged.
