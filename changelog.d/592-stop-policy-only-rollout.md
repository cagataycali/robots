### Fixed: a bare `stop_policy` stops the only rollout in flight

Every "Stop it first: action='stop_policy'" remedy now works as written:
with exactly one policy running, `stop_policy` without `robot_name` stops
it. With several running the refusal names them; with none it says so and
names the robots. The per-robot remedies spell the parameter `robot_name`.
