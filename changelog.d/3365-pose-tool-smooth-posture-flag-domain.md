### Fixed: `pose_tool`'s `smooth` is checked, not read by truthiness

`smooth` selects one of two trajectories towards the same joint targets -
interpolate over `steps * step_delay` seconds, or write each `Goal_Position`
once - and it was read by truthiness, so both undeclared halves silently
inverted. `smooth=0` (also `""`, `None`, `[]`) wrote 2 goal positions where
`smooth=True` writes 42 over 21 increments; since the flag defaults to `True`,
that *removes* an interpolation the caller never asked to leave and sends the
arm to the far end of its travel in one write - the full-travel jump the tool
already refuses `steps=True` for. `smooth="false"` (also `"no"`, `"off"`,
`"0"`) is truthy, so it kept the interpolation the word asks to skip. Because
the flag decides whether `steps` / `step_delay` are read at all,
`smooth="false", steps=0` was refused for `steps`, an option the caller's
posture said nobody would read.

The flag is now held to the shared `boolean_flag_error` domain, ahead of the
`steps` / `step_delay` check so a bad flag is named as the flag, and only for
`"load_pose"` and `"move_multiple"` - `"reset_to_home"` supplies its own
`smooth=True` and every other action moves in one shot, so none of them is
refused for it. Both declared postures are unchanged.
