### Fixed: an interpolated pose move reports the joints it actually commanded

`MotorController._smooth_move` ended in a literal `return True` and threw away
every `move_motor` result, so the interpolating path answered success whatever
reached the bus. Two joints could go missing silently. A motor whose current
position did not arrive has no start point, so no trajectory is built for it and
the write loop never commands it - it was dropped from the move without a word.
And a write the bus refused was logged by `move_motor` and then discarded by the
loop.

Every other commanding method on the class already answered for what it did.
`move_multiple_motors(smooth=False)` reads each `move_motor` outcome,
`disable_torque` returns the motors it could not reach - its docstring explains
why, that a caller told the whole arm was released when part of it is still
driven is worse than no stop at all - and `incremental_move` refuses outright
when the current position is unreadable. So one method carried two contracts a
single `smooth` flag apart, and `smooth` defaults to `True`: the default path was
the one that could report a pose it had not commanded a single packet towards.
`reset_to_home` passes `smooth=True` itself and had no honest branch to take,
which made the action an operator reaches for to put the arm somewhere known the
one that could not decline to interpolate.

On a bus that accepts writes and answers no position read - an ordinary
half-duplex symptom - `pose_tool(action="move_multiple")` listed both requested
angles under `status="success"` with zero `Goal_Position` packets written, while
the same call with `smooth=False` on the same bus wrote both and reported
honestly. The interpolating branch now returns `False` when a joint was left
uncommanded or a write failed, and logs which joints at ERROR, so the tool's own
handlers turn it into an error envelope the way they already do for the one-shot
branch.

Every motor is still attempted, matching `disable_torque` and the one-shot
branch: the reported outcome changed, the packets did not. A step count that runs
no increments is deliberately still a success - `_smooth_move_option_error`
refuses `steps <= 0` before any caller of `pose_tool` reaches the loop, and
asking for no increments is not the same failure as a joint that could not move.

The suite could not see any of this. `reset_to_home`'s existing ASCII test drove a
serial double that answers no read, so it asserted `status="success"` on a call
that wrote nothing; it now uses the answering double, promoted into the shared
`tests/tools` conftest that a third module needed, and asserts the bus was really
driven.
