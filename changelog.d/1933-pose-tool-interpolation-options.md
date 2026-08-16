### Fixed: `pose_tool` honors its `steps` / `step_delay` interpolation options instead of dropping them

`pose_tool`'s documented `steps` and `step_delay` parameters never reached the
interpolation loop that reads them. `move_multiple_motors` did not accept them and
called `_smooth_move(positions)` with neither, so every interpolated move -
`load_pose`, `move_multiple` and `reset_to_home` - ran the hardcoded 20 increments
at 0.05s regardless of what the caller asked for, and reported success. An agent
asking for a slower, finer trajectory on a real arm silently got the default one.

Both options are now forwarded, and validated at the tool boundary before any pose
file is read or the serial port is opened, because each is consumed inside the loop
on a live servo bus: `steps` is the divisor for each motor's increment and the bound
of the write loop (`positive_count_error`), and `step_delay` is the pause between
goal positions (`positive_finite_number_error`). Previously unusable values would
have raised `ZeroDivisionError`, silently skipped the loop and reported a move that
never happened, jumped at full travel in a single increment, or - for an infinite
delay - blocked forever with the arm stopped part-way through its trajectory and the
port still open. Only the actions that interpolate are checked; any other action
ignores both options and is never refused for them.
