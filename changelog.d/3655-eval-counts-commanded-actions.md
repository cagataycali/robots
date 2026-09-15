### Fixed: an evaluation whose every action commands nothing is refused, not scored

`eval_policy` reports `actions_applied` beside `steps_advanced` and refuses the
evaluation when the policy never commanded the robot, because the published
`success_rate` / `avg_reward` / `pass_hat_k` then describe the scene's initial
state. That counter incremented once per call to `send_action`, and an action
dict with **no keys** reaches `send_action` like any other: the backend accepts
it and commands no actuator. A chunk of such actions is not empty, so the
empty-chunk guard never saw it, and the counter reported one applied action per
step.

Measured on a MuJoCo SO-101, a policy returning `[{}, {}, ...]` and a policy
commanding all six joints reported the same `actions_applied` (60) under the same
`status="success"`, with 0 versus 360 keys reaching `send_action`. The sibling
`run_policy` surface already separated the two through its per-actuator
`action_resolution_rate` (`partial_action_failure_rate` 1.0 versus 0.0); only the
evaluation routes read a counter that could not.

`actions_applied` now counts actions that commanded at least one key, stated once
as `action_commands_robot` and read by all three evaluation loops (synchronous,
async-RTC, and the benchmark-spec route). An all-uncommanding evaluation is
refused with the existing message; a *partial* shortfall stays a reported count,
so the per-step tolerance granted to an empty chunk is unchanged, and it is now
visible at all (`actions_applied` was previously equal to `steps_advanced` for a
policy commanding on half its actions). `CuroboPolicy._next_chunk` emits one of
these actions per waypoint when the planner's trajectory rows carry no joint
position: the key list it zips against is empty too, so every waypoint decodes to
an action naming no key while the chunk itself stays non-empty.
