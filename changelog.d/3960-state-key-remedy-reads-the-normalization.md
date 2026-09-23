### Fixed

- `lerobot_local`: the state-key mismatch remedy no longer recommends a
  unit-converting `embodiment=` to a caller whose normalization is inert.
  `so100` / `so101` declare `state_units='degrees'`, correct only against
  degree-recorded stats; with a base checkpoint's dataset-keyed stats nothing
  scales the conversion back, so the so101 joint range reached the model at up
  to 160.0 where packing the observation's own keys reaches 2.79. The remedy now
  reads `ProcessorBridge.inert_normalization_features()`, points at
  `set_robot_state_keys([...])`, and names the `processor_overrides` that would
  make the embodiment correct. A native-units candidate is unaffected.
