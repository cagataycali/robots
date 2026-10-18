### Fixed: the twin shows what the arm measured; a closed Simulation refuses; the public types are values

`real.observe()` used to write the measured joints into `real.twin` as targets only, so the twin stood at zero
until its world had stepped a few hundred times and `real.twin.sim.render()` drew the wrong pose. It now places
the twin's joints at the measurement (`Backend.set_joints`: positions set, velocities zero, no step); an engine
that cannot place joints keeps the command-only mirror. `add`, `add_object`, `step`, `control_step`, `reset` and
`run` on a closed `Simulation` raise `RuntimeError` instead of driving a closed backend. `JointState`, `Pose`,
`Observation` and `RunResult` compare by value (`==` compares their arrays, for any number of joints) and hash
consistently, so they work in `assert`, sets and dict keys.
