### Fixed: a fleet-wide sim stop reports a rollout whose `stop_policy` refused, instead of answering `ok=True` over it

`Mesh.emergency_stop` broadcasts `{"action": "stop"}` and carries no `robot_name`, so on a
simulation peer it lands in the fleet-wide branch of `_dispatch`: read every active rollout
from `_active_policy_robots()` and call `stop_policy` once per robot. That branch is the
only stop path that aggregates, and it aggregated by assuming the answer -- the per-robot
results were collected into `results` and the envelope hardcoded `ok=True`.

`_peers_that_did_not_stop` reads only the top-level `ok`/`status`, never `results`, so a
`stop_policy` that refused was scored as a halt: the refusal sat in the payload unread and
the robot was listed under `stopped`. `emergency_stop` then kept the peer out of
`peers_not_stopped` and did not fire the CRITICAL "robots may still be executing" warning.
That is the affirmative lie the two sibling stop branches are commented against, and the
exact shape `_peers_that_did_not_stop`'s own docstring lists as one it must catch.

The two paths therefore disagreed about one refusal. Asked to stop `bob` by name the peer
answered `{"status": "error", ...}` and was flagged; asked to stop everything with `bob`
among the rollouts it answered `{"ok": True, "stopped": ["alice", "bob"], ...}` and was not.
The path the fanout uses is the one that swallowed it.

`ok` is now derived from the per-robot answers rather than assumed, through a single shared
predicate. `_reports_failure_to_stop` owns the rule `ok is False or status == "error"` --
both spellings, because the stop verbs disagree about their envelope: `stop_task` and the
dispatch's own branches answer `{"ok": ...}` while `Simulation.stop_policy` answers the
agent-tool `{"status": ...}`. `_peers_that_did_not_stop` reads that predicate instead of its
own copy of the comparison, since a second copy of the rule is how the branch came to
contradict the aggregation. The fleet-wide branch grades each answer with it, reports
`ok=False` with `not_stopped` naming the refusing robots, keeps `stopped` for the ones that
really stopped, logs at ERROR, and leaves `results` in place so the detail is available.

A fleet stop in which every rollout stopped is unchanged, with no `not_stopped` and no
`error` key; a peer with no rollouts and a robot-less gateway peer are both untouched.
