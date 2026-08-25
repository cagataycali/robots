### Fixed: a sim peer's state topic reports which robot is running a policy

`Mesh._read_state` publishes a `robots` section naming every robot in the sim world, and the flag
beside each name was the literal `True`. So it was constant for the life of the peer: it did not
change when a policy started, it did not change when one stopped, and a scene's idle arms were
indistinguishable from the one arm executing a rollout. Measured on a two-arm MuJoCo world with a
mock rollout on `so100` only, reading the topic in all three phases:

    at rest            {"so100": {"active": true}, "arm_b": {"active": true}}
    rollout on so100   {"so100": {"active": true}, "arm_b": {"active": true}}
    after stop         {"so100": {"active": true}, "arm_b": {"active": true}}

The `status` command answers the same question correctly over the same three phases -
`robots_running` is `[]`, `["so100"]`, `[]` - because it reads the running-policy registry. Two
surfaces of one peer therefore disagreed about one fact, and the 10 Hz telemetry topic was the one
that was wrong. The flag is now read from that same registry, so the topic and an on-demand status
answer always name the same set.

That disagreement was already known when the `status` path was written. Its own comment says sims
"answered `{"status": "unknown"}` and their state topic hardcoded active=True - a running sim
policy was invisible on the wire", so the state topic was named as part of the defect and only the
command half was repaired. Nothing graded the remaining half: no test in `tests/mesh/` asserted on
the flag at all, and `test_mesh.py::test_publishes_sim_clock` asserts the section's key set
(`sorted(s["robots"].keys()) == ["arm0", "arm1"]`) without reading a value, which is exactly the
shape that lets a constant survive.

A `SimRobot` child peer keeps no registry of its own, so it consults the parent `Simulation`
through the `_sim_parent` backref - the same backref `_dispatch` already uses to delegate
`execute`/`start`, installed by `Simulation._attach_robot_to_mesh` alongside the `_world`
reference that makes the child publish a state topic in the first place. Both peer kinds now
report the identical map.

Two dispositions are deliberate. A peer that keeps no such registry reports every robot `false`:
nothing runs a policy on those robots through the simulation API, so none of them is executing
one, and its `robots` section is still published. A registry that raises is left to reach
`_read_state`'s section handler, which names `sim_world` in `degraded` - the documented mechanism
for a probe that cannot answer. That is the opposite of `status`, which swallows because a command
must answer; substituting a flag on the telemetry path is what would make the fault unreportable,
and it is how the unmeasured `True` survived in the first place.

The flag distinguishes a rollout from motion rather than motion from stillness, which is why it
is read from the registry and not from the world. In the measured scene `so100` moves 0.4585 rad
under its policy while `arm_b` moves 0.0320 rad - it is sagging under gravity, not executing
anything - and a motion heuristic would have called both active.

`docs/mesh.md` documented neither the section nor the flag; the published-topics table listed the
state topic's content as "joints, sim time, task status, degraded probes". It now documents what
`active` means, that it shares its source with `robots_running`, that which robots *exist* is the
presence topic's `sim_robots`, and both boundary dispositions above.
