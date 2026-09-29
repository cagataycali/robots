### Fixed

- **policies/cosmos3**: the flat `observation.state` vector is read as the
  `joint_pos` state row (7 joints then the gripper), as the docs page already
  said and every other provider does; it was refused as "found 0". An action
  chunk whose width is not the active layout's is now refused, naming the
  layout and the `action_space` the server must have been launched with; it
  used to be named positionally and padded `action_<i>`, which handed a
  `midtrain` server's pose columns to the robot as joint targets. An
  `observation_mapping` target outside the `observation/` namespace is refused
  at construction instead of being skipped without a word. (#4194)
