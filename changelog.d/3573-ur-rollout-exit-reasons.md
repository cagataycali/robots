### Tests: the UR rollout's exit reasons and both connect failures are graded

`tests/drivers/ur/test_ur_driver_over_a_fake_controller.py` drove the paths where
the arm answers. Four of `_Rollout._run`'s exits and both halves of
`URDriver.connect_eagerly`'s failure ladder were graded by nothing: 43 of
`drivers/ur.py`'s 459 statements uncovered.

`exit_reason` is what an operator reads to learn why an arm stopped moving, so the
ungraded set was the half that matters most. Six cells over it, in the file that
already owns this driver's controller double:

* `duration` -- a rollout with no step budget ends on its clock, and reports that
  rather than a refusal (`refusal` stays `None`).
* `policy` from a *raised* step, distinct from the already-graded "returned no
  action dict": the exception class and its message identify the fault, and
  nothing reaches `servoJ`.
* `stopped` before the first step, and `stopped` at the pacing tick after a stop
  landed *inside* a write -- the setpoint already on the wire is counted, and the
  tick answers the stop instead of commanding another step. The second is armed
  from the rollout thread through the double's mode-read hook, so it lands in the
  window the loop's comment describes rather than near it.
* A controller that does not answer RTDE receive is reported by address, and the
  control side is not dialled at all.
* A control side that refuses releases the receive interface it opened. The driver
  records `_receive` only once both sides open, so an interface left connected
  there is unreachable afterwards -- `cleanup` releases what was recorded.

Measured by mutation: six changes to the behaviours these cells grade, five
detected singly. The sixth -- deleting the loop's top-of-loop stop read -- is
absorbed by the pre-write re-read, which is by design (that comment says the
re-read "cannot be the only one"); removing both fails the cell on `steps`.

`drivers/ur.py`: 43 -> 27 missing, 91% -> 94%. No production statement changed.
