### Fixed: a unit frame the pack-state pipeline step cannot convert is refused

`EmbodimentMap` refuses a `state_units` outside the closed `UNIT_FRAMES`
vocabulary, but the `strands_pack_state` pipeline step holds the same field and
compares it against `"degrees"` the same way, and graded nothing. The map is not
its only source: LeRobot's `from_pretrained` rebuilds the step from the
`policy_preprocessor.json` a checkpoint ships, so a saved config naming
`"DEGREES"`, `"deg"` or `"radians"` built a step that silently meant `"native"`
and packed the sim's raw radians (`0.5`) where `"degrees"` packs `28.648`, for a
model trained on degrees. Both holders now call one refusal, so a reconstructed
step is graded like the map that normally fills it.
