### Fixed: a reloaded processor pipeline keeps the packed state's unit frame

`PackStateProcessorStep.get_config()` emitted 3 of the 8 fields the step reads at
runtime, and LeRobot rehydrates a registered step from exactly that dict. A
saved-and-reloaded `state_units="degrees"` frame therefore came back as
`"native"` and packed the sim's raw radians where the checkpoint was trained on
mid-centered degrees -- 0.5 rad reached the model as 0.5 instead of 28.6 degrees,
with no warning -- and `strict_keys=True` came back `False`, so a declared key the
observation did not carry was zero-filled instead of refused. Every declared field
is now emitted except `missing_keys_sink`, which is the policy's live list rather
than configuration.
