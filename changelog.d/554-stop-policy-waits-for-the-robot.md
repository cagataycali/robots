### Fixed: `stop_policy` answers once the robot is free

`stop_policy` lowered the cooperative flag and returned `Stopped` while the
worker was still winding down. The caller's very next `start_policy` (or any
joint write) on the same robot was then refused `while its policy is running`,
and a second `stop_policy` in that window answered `Was not running` against a
`list_policies_running` that still listed the robot.

MuJoCo's `stop_policy` now joins the `start_policy` Future it just flagged,
bounded by `_POLICY_STOP_JOIN_TIMEOUT` (1.0 s; a healthy rollout exits within
one control tick, so the wait is normally a few ms). The `json` block gains
`exited`: `True` when the worker is gone, `False` when it is still live after
the bound (the text says so and names the consequence), `None` when there was
nothing to join - no rollout, or a blocking `run_policy` driven on its
caller's thread.
