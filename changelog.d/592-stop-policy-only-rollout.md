### Fixed: a bare `stop_policy` stops the only rollout in flight

Every "Stop it first: action='stop_policy'" remedy now works as written:
with exactly one policy running, `stop_policy` without `robot_name` stops
it. With several running the refusal names them; with none it says so and
names the robots. Every gate's remedy is a call the tool accepts: the
per-robot ones spell the parameter `robot_name`, and a gate offers the bare
form only where it resolves, naming each robot when several are in flight.
