### Fixed: `stop_policy` answers once the robot is free

`stop_policy` lowered the cooperative flag and returned `Stopped` while the
worker was still winding down, so the caller's very next `start_policy` (or any
joint write) on the same robot was refused `while its policy is running` -
measured on 40 consecutive stop-then-start pairs, all 40 refused. A second
`stop_policy` in that window answered `Stopped` again rather than
`Was not running`, so neither answer told the caller the robot was free.

MuJoCo's `stop_policy` now joins the `start_policy` Future it just flagged,
bounded by `_POLICY_STOP_JOIN_TIMEOUT` (1.0 s; a healthy rollout exits within
one control tick, so the wait is normally a few ms). The `json` block gains
`exited`: `True` when the worker is gone, `False` when it is still live after
the bound (the text says so and names the consequence), `None` when there was
nothing to join - no rollout, or a blocking `run_policy` driven on its
caller's thread.

Because that wait makes a SEQUENTIAL fanout over `stop_policy` pay it between
one robot's stop request and the next robot's, the mesh `{"action": "stop"}`
branch now lowers every target's flag before it joins any of them - the same
order the engine's own teardown uses. Without the pre-pass, one robot whose
policy server was wedged inside inference held the stop off the robots behind
it: on a three-robot world the two healthy workers exited at t+1.007s and
t+1.011s instead of t+0.004s and t+0.009s, still driving their arms until then.
