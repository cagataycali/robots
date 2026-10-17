### Added: agent tools generated from the manifest

`robot.tool` is a strands agent tool built from the robot's manifest: its rpcs (observe, act, run, stop, estop,
resume, ...) and their signatures become the tool spec, so a new rpc is a new tool action with nothing hand-written.
`sim.tool` does the same for a world's public verbs (add a robot, an object or a camera, step, reset, render). A
command from a tool is an agent's: it goes through the safety chain and the operator gate on every step, including
`run` and `start` in a background thread. strands is imported only when a tool is built (`[tools]` extra). A policy
spec string is accepted wherever a policy is.

`describe()` reports where objects are now and each robot's live `pose`; `pose(body)` reads one. Vectors on the
wire are fixed-length number arrays (a string colour is refused by name), `randomize` takes a `(lo, hi)` range and
returns what it drew, `reset(keyframe=)` refuses an unknown keyframe, `remove` takes an object or a camera, and
`Simulation.run({"a": spec, "b": spec})` drives robots in lock-step from the tool.
