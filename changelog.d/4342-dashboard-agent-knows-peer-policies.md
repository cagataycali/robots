### Added: the dashboard agent knows which policies each robot on the mesh can run

Every `fleet` row carries `policies`: the robot the peer is (read from a sim
child's id, a one-robot sim world, the managed table or the driver name) and
`can_run`, the registry providers that apply to it with the kwargs the wire
carries, typed and bounded, what each requires, and one line on what it does
(`strands_robots.dashboard.peer_policies`). Embodiment-bound providers (`wbc`,
`wbc_gait`, `microduck`) appear only on their robot, and first; `mock` last;
removed and server-only providers are not offered. The system prompt says to
pick from that list, to use `mock` only when the operator asks, and to ask for
a missing checkpoint or server instead of guessing.

The wire carries what a whole-body controller needs, fail closed: `walk` is
accepted by `mesh.security.validate_command` as a strict boolean only, both
dashboard proxy rails forward `model_path`, `walk` and `target_velocity` on
`execute` / `start` (the envelope bound on `target_velocity` was already
there), and `create_policy` spells the wire's `model_path` as the `checkpoint`
a provider declares (`wbc`, `wbc_gait`) instead of dropping it. Measured on a
sim Unitree G1: one operator message to the dashboard agent runs `wbc` and the
robot walks 1.89 m in 5 s. `config` is not carried on purpose.
