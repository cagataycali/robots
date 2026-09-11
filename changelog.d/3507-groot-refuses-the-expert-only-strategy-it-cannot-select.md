### Fixed: GR00T refuses a tuning strategy it cannot select

`Gr00tTrainer.validate()` accepted `method="expert_only"` and forwarded it
nowhere. GR00T chooses what trains through its `tune_*` switches, and nothing
resolved that strategy into them, so the request produced no problems and the
launch argv was byte-identical to `method="full"` - including
`--tune_projector=true` for a caller who asked to train the action expert alone.
With `tune={"llm": True}` it emitted `--tune_llm=true`, fine-tuning the language
model under an expert-only label and reporting success.

`expert_only` belongs to the LeRobot policies whose config declares
`train_expert_only` (pi0 / pi05 / smolvla); `GrootConfig` declares the `tune_*`
switches instead, which is why `LerobotTrainer(policy_type="groot")` already
refused the same word for the same architecture. This trainer now agrees with
it, and the refusal names the switch that does express the request:
`tune={"projector": False}` leaves only the diffusion action head training.
`full` and `frozen_backbone` are unchanged.
