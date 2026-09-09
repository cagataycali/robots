### Fixed

- **policies/protomotions**: `ProtoMotionsConfig.action_ema_alpha` is now applied
  to the joint targets the PD loop receives, and is refused unless it is a finite
  number in `(0, 1]`. The field was parsed from the checkpoint's
  `unified_pipeline.yaml`, stored on the config and read by nothing, so the
  emitted targets were byte-identical for every declared factor; having no reader
  it also had no domain, so `0` (which freezes the commanded pose at the first
  tick's target), a negative weight, a value above `1` and `nan` were all stored
  and carried into the control path. `1.0` - the shipped checkpoint's own value -
  remains bit-exact passthrough, the first tick of an episode seeds the filter
  from the network's own output rather than from zeros, and the historical-actions
  buffer keeps carrying the raw output because the graph's
  `historical_processed_actions` input is defined over it.
