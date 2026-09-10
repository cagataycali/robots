### Changed: the `groot` training provider is `LerobotTrainer(policy_type="groot")`

`create_trainer("groot")` post-trains GR00T through lerobot instead of an
out-of-tree Isaac-GR00T checkout, so a GR00T fine-tune needs one install
(`pip install 'strands-robots[lerobot]' 'lerobot[training]'`), no `GR00T_ROOT`,
and gets the same resume, LoRA, sample-weighting and validation path as every
other policy. A provider's `trainer` block in `policies.json` may now declare
`defaults` - constructor kwargs a caller can still override - which is how one
trainer class serves a provider without a subclass whose only content is a
preset. `method="frozen_backbone"` is no longer accepted for GR00T; spell it
`tune={"llm": False, "visual": False}`.
