### Fixed

- **policies/protomotions**: a batched `observation.state` now resolves through
  every observation convention `ProtoMotionsPolicy` accepts. `_pack_by_name`
  reads that one key from two of its three conventions - paired with a
  `state_keys` list supplied on the observation, and paired with the policy's own
  `_robot_state_keys` - and both index the array positionally, but only the
  second flattened it first. A `(1, D)` state, the shape LeRobot's
  `AddBatchDimensionObservationStep` produces, therefore resolved without the key
  list and died with it: `get_actions` raised `TypeError: only length-1 arrays
  can be converted to Python scalars`, a message naming neither the key, the
  shape nor a remedy, and a type outside its own documented `Raises`. The
  flattening now happens once, ahead of every convention, so the two readers
  cannot disagree about the same observation.
