### Fixed: supplied normalization stats normalize a pi checkpoint, and pi0-FAST loads with its own stats

`processor_overrides={"normalizer_processor": {"stats": ...}}` - the documented remedy
for missing or inert stats - normalized nothing on `lerobot/pi05_base` and
`lerobot/pi05_droid`: their `policy_preprocessor.json` declares `features: {}`, and a
normalizer only touches the features it declares, so a -2.2 rad joint reached
pi0.5's 256-bin state tokenizer unchanged. A stats override without `features` now
gets them from the policy config, as `lerobot-train` builds the steps (the input and
output features for the normalizer, the output features for the unnormalizer,
`norm_map` from `normalization_mapping`); a caller's own `features` / `norm_map` are
kept (`complete_stats_overrides`). `ProcessorBridge.inert_normalization_features`
now reports a normalizer that declares no features, so the existing inert-stats
warning and refusal fire for those checkpoints. And `lerobot/pi0fast-libero` was
refused with its OWN stats: it declares `observation.state` at `max_state_dim` (32)
and ships 8-wide state stats, which the width guard read as a mismatch. For a model
that declares `max_state_dim`, narrower state stats are widened to the padded width
with neutral values (the padded zeros normalize to zeros), and any other mismatch is
still refused.
