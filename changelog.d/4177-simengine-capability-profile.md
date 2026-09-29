### Added: `SimEngine.capabilities()` names what a backend supports

`SimEngine` has about thirty abstract members, most of them
manipulation-centric (`add_object`, `robot_joint_names`, `render`, ...), and
nothing told a caller which of them a given backend actually honours.

`strands_robots.simulation.capabilities` now names them: four core capabilities
(`world`, `robots`, `step`, `observation`), four that complete the default set
(`joints`, `objects`, `render`, `policy_rollout`), and six optional ones
credited when the backend overrides the method behind them (`load_scene`,
`randomize`, `obs_noise`, `contacts`, `frames`, `camera_params`).
`sim.capabilities()` returns the set and `describe()["capabilities"]` reports
it. A backend may declare the set as a `CAPABILITIES` class attribute, validated
at class creation: the core four are required, a name outside the vocabulary
must be spelled `vendor:name`, and an optional name needs its method
overridden. A backend that declares nothing reports exactly what it implements,
so every built-in backend's behaviour is unchanged. `check_capabilities(sim,
required, caller=...)` returns an `unsupported_by_backend` error result naming
what is missing, for a caller to check before it calls.
