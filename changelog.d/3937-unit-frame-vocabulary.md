### Fixed: a unit frame an embodiment map cannot convert is refused

`EmbodimentMap.state_units` / `action_units` are plain strings, and every
conversion site compares them against `"degrees"`, so any other spelling meant
`"native"` with no conversion and no warning -- including `"DEGREES"`, the
spelling LeRobot's own `MotorNormMode` uses. On a degrees-trained SO-arm
checkpoint that packs the sim's raw radians as `observation.state` and feeds raw
degree actions back into radian joint limits. The two frames are now a closed
vocabulary (`UNIT_FRAMES`) refused at construction, the way the sibling
`dim_policy` field has always been refused by `reconcile_dim`, so
`embodiments.json` and an inline `load_embodiment({...})` dict are both graded.
