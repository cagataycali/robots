### Fixed: every simulation backend reports the rollouts it is driving

The mesh answers "is a rollout in flight" on two surfaces - the `status`
command and the `robots` section of the state topic - and both probed for
`_active_policy_robots`, a method only the MuJoCo engine defines. A Newton
peer with a rollout genuinely in flight was invisible on both, while the
stop verb on that same peer halted it:

| peer | rollout | `status` | state topic | fleet `stop` |
|---|---|---|---|---|
| Newton | in flight | `{"status": "unknown"}` | `{"arm": {"active": false}}` | halts it |
| Newton | idle | `{"status": "unknown"}` | `{"arm": {"active": false}}` | nothing to halt |
| Isaac | in flight | `{"status": "unknown"}` | no section | refuses |
| MuJoCo | in flight | `running`, `["arm"]` | `{"active": true}` | halts it |

The two Newton rows were identical, so a caller could not tell a live
rollout from an idle world. The state topic's row is the worse of the two:
`active=false` is an affirmative "this robot is idle", published on no
evidence, about a robot the peer was in fact driving.

Both backends keep the per-robot `policy_running` claim already, and Isaac
already trusts it as a population for `run_multi_policy`'s busy guard - what
was missing was a shared way to ask. `SimEngine._rollouts_in_flight` is that
seam, tri-state in the direction `_request_policy_stop` already is: the
names when the backend can enumerate its rollouts, and `None` when it keeps
no such claim, which is a stated absence of a verdict rather than an empty
population. Reporting a rollout is a strictly weaker requirement than
halting one, which is why Isaac answers it while `stop_policy` still refuses
there: a bare flag is not a durable claim to move (#2833), but it is a true
answer to what is running.

Newton and Isaac override it off the claim they maintain; MuJoCo delegates to
`_active_policy_robots` so the union of the Future table and the per-robot
flag stays spelled once. Both mesh surfaces now read one call, so a child
`SimRobot` peer's `status` answers for the world exactly as its state topic
already did. A peer reporting no population has its robots named with **no
`active` key** and answers `status="unknown"`: absence is how every other
section of that snapshot spells "the probe had nothing", where `false` was
indistinguishable from a rollout the peer cannot see.

The fleet stop's population is deliberately NOT this population. It asks
every robot in the world for a backend keeping no pruned rollout registry -
`stop_policy` is idempotent and reports `was_running` itself, and a blocking
`run_policy` registers no future - so narrowing it to the in-flight set is a
decision about a safety path (#3359), not part of reporting. Two cases pin
that it did not move.

`tests/mesh/test_every_sim_backend_reports_its_rollouts_in_flight.py` pins
both phases on both backends, the tri-state's three answers, the two
surfaces agreeing from one call, and the unchanged stop population; 11 of
its 15 cases fail on the previous code.
